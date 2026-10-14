import gc
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from huggingface_hub import model_info
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def parallel(model):
    return torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model


class Classical:
    def __init__(self, recipe, config, output):
        self.recipe = recipe
        self.config = config
        self.output = Path(output)

    def fit(self, train, validation):
        vector = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True, strip_accents='unicode')
        if self.recipe['family'] == 'logistic':
            self.model = Pipeline([('tfidf', vector), ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=self.config['seed']))])
        else:
            folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=self.config['seed']).split(train['review_text'], train['target'], train['duplicate_group']))
            estimator = Pipeline([('tfidf', vector), ('classifier', LinearSVC(class_weight='balanced', random_state=self.config['seed']))])
            self.model = CalibratedClassifierCV(estimator, cv=folds, method='sigmoid', n_jobs=1)
        self.model.fit(train['review_text'], train['target'])

    def predict(self, frame):
        return self.model.predict_proba(frame['review_text'])

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.output / 'pipeline.joblib')
        return {'family': self.recipe['family'], 'path': 'pipeline.joblib'}


class TextRows(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        encoded = self.tokenizer(row.review_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {**{key: value.squeeze(0) for key, value in encoded.items()}, 'target': torch.tensor(int(row.target))}


class CompactTransformer:
    def __init__(self, recipe, config, output):
        self.recipe = recipe
        self.config = config
        self.output = Path(output)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.revision = recipe.get('revision') or model_info(recipe['checkpoint']).sha
        self.tokenizer = AutoTokenizer.from_pretrained(recipe['checkpoint'], revision=self.revision)
        self.model = parallel(AutoModelForSequenceClassification.from_pretrained(recipe['checkpoint'], revision=self.revision, num_labels=len(config['classes'])).to(self.device))

    def fit(self, train, validation):
        self.output.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(TextRows(train, self.tokenizer, self.recipe['max_length']), batch_size=self.recipe['batch_size'], shuffle=True, num_workers=0)
        counts = np.bincount(train['target'], minlength=len(self.config['classes']))
        weights = torch.tensor(len(train) / (len(counts) * counts), dtype=torch.float32, device=self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.recipe['learning_rate'], weight_decay=0.01)
        scaler = torch.amp.GradScaler('cuda', enabled=self.device == 'cuda')
        best, stale, history = -1.0, 0, []
        for epoch in range(self.recipe['epochs']):
            self.model.train()
            losses = []
            for batch in loader:
                targets = batch.pop('target').to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.device == 'cuda'):
                    logits = self.model(**{key: value.to(self.device) for key, value in batch.items()}).logits
                    loss = criterion(logits.float(), targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            scores = self.predict(validation)
            value = f1_score(validation['target'], scores.argmax(1), average='macro', zero_division=0)
            history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'validation_macro_f1': value})
            print('TRANSFORMER_EPOCH', json.dumps(history[-1]), flush=True)
            if value > best:
                best, stale = value, 0
                unwrap(self.model).save_pretrained(self.output / 'checkpoint', safe_serialization=True)
                self.tokenizer.save_pretrained(self.output / 'checkpoint')
            else:
                stale += 1
                if stale >= self.recipe['patience']:
                    break
        (self.output / 'training_history.json').write_text(json.dumps(history, indent=2))
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = parallel(AutoModelForSequenceClassification.from_pretrained(self.output / 'checkpoint').to(self.device))

    def predict(self, frame):
        self.model.eval()
        scores = []
        for start in range(0, len(frame), self.recipe['batch_size']):
            inputs = self.tokenizer(frame['review_text'].iloc[start:start + self.recipe['batch_size']].tolist(), padding=True, truncation=True, max_length=self.recipe['max_length'], return_tensors='pt').to(self.device)
            with torch.inference_mode():
                logits = self.model(**inputs).logits.float().softmax(-1).cpu().numpy()
            scores.append(logits)
        return np.concatenate(scores)

    def save(self):
        return {'family': 'transformer', 'checkpoint': self.recipe['checkpoint'], 'revision': self.revision, 'path': 'checkpoint'}


class Generative:
    def __init__(self, recipe, config, output):
        self.recipe = recipe
        self.config = config
        self.output = Path(output)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.revision = recipe.get('revision') or model_info(recipe['checkpoint']).sha
        self.tokenizer = AutoTokenizer.from_pretrained(recipe['checkpoint'], revision=self.revision, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = parallel(AutoModelForCausalLM.from_pretrained(recipe['checkpoint'], revision=self.revision, dtype=torch.float16 if self.device == 'cuda' else torch.float32, attn_implementation='sdpa').to(self.device).eval())
        self.prompts = json.loads((Path(__file__).parent / 'experiment_prompts.json').read_text())
        encoded = [self.tokenizer.encode(name, add_special_tokens=False) for name in config['classes']]
        assert all(len(value) == 1 for value in encoded), 'VERBALIZER_TOKEN_LENGTH'
        self.class_tokens = [value[0] for value in encoded]
        self.trace = []

    def fit(self, train, validation):
        self.train = train.reset_index(drop=True)
        if self.recipe['family'] == 'agent':
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)
            self.retrieval_matrix = self.vectorizer.fit_transform(self.train['review_text'])

    def score(self, texts, examples=None):
        inputs = []
        for index, text in enumerate(texts):
            content = {'review': text[:1400]}
            if examples is not None:
                content['examples'] = examples[index]
            messages = [{'role': 'system', 'content': self.prompts['sentiment']}, {'role': 'user', 'content': json.dumps(content, ensure_ascii=False)}]
            inputs.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        encoded = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.recipe['max_length'], return_tensors='pt').to(self.device)
        with torch.inference_mode():
            logits = self.model(**encoded).logits[:, -1, self.class_tokens].float()
        return logits.softmax(-1).cpu().numpy()

    def tool(self, text, score):
        messages = [{'role': 'system', 'content': self.prompts['tool']}, {'role': 'user', 'content': json.dumps({'review': text[:1000], 'initial_sentiment': self.config['classes'][int(score.argmax())], 'score': round(float(score.max()), 3)})}]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = unwrap(self.model).generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        decision = self.tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().lower()
        return {'action': decision if decision in ['retrieve', 'finalize'] else 'finalize', 'valid': decision in ['retrieve', 'finalize'], 'raw': decision}

    def predict(self, frame):
        scores = []
        for start in range(0, len(frame), self.recipe['batch_size']):
            part = frame.iloc[start:start + self.recipe['batch_size']]
            initial = self.score(part['review_text'].tolist())
            if self.recipe['family'] == 'agent':
                for local, row in enumerate(part.itertuples()):
                    before = initial[local].copy()
                    decision = self.tool(row.review_text, before)
                    action = decision['action']
                    ids = []
                    if action == 'retrieve':
                        similarity = (self.vectorizer.transform([row.review_text]) @ self.retrieval_matrix.T).toarray().ravel()
                        order = np.argsort(-similarity)
                        selected = [i for i in order if self.train.iloc[i]['duplicate_group'] != row.duplicate_group and similarity[i] > 0][:2]
                        examples = [{'review': self.train.iloc[i]['review_text'][:300], 'sentiment': self.train.iloc[i]['reviewer_sentiment']} for i in selected]
                        ids = [self.train.iloc[i]['review_id'] for i in selected]
                        if examples:
                            initial[local] = self.score([row.review_text], [examples])[0]
                    self.trace.append({'review_id': row.review_id, 'tool': action, 'tool_output_valid': decision['valid'], 'tool_output': decision['raw'], 'evidence_ids': ids, 'initial_class': int(before.argmax()), 'final_class': int(initial[local].argmax()), 'initial_score': float(before.max()), 'final_score': float(initial[local].max()), 'review_needed': bool(not decision['valid'] or before.argmax() != initial[local].argmax() or initial[local].max() < 0.55)})
            scores.append(initial)
        return np.concatenate(scores)

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(self.output / 'tokenizer')
        if self.recipe['family'] == 'agent':
            joblib.dump({'vectorizer': self.vectorizer, 'matrix': self.retrieval_matrix, 'training_ids': self.train['review_id'].tolist(), 'training_groups': self.train['duplicate_group'].tolist()}, self.output / 'retriever.joblib')
            self.train[['review_id', 'review_text', 'duplicate_group', 'reviewer_sentiment']].to_parquet(self.output / 'retrieval_examples.parquet', index=False)
        (self.output / 'tool_trace.json').write_text(json.dumps(self.trace, indent=2), encoding='utf-8')
        return {'family': self.recipe['family'], 'checkpoint': self.recipe['checkpoint'], 'revision': self.revision, 'dtype': 'float16' if self.device == 'cuda' else 'float32', 'class_tokens': self.class_tokens, 'weights': 'pinned_hub_revision'}


def create(recipe, config, output):
    seed_all(config['seed'])
    if recipe['family'] in ['logistic', 'svm']:
        return Classical(recipe, config, output)
    if recipe['family'] == 'transformer':
        return CompactTransformer(recipe, config, output)
    return Generative(recipe, config, output)
