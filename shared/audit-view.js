import {escapeHTML as e,title,aspectNames} from './core.js?v=20261016-audit1';

export const recommendationNames={resolve_ambiguity:'Resolve ambiguity',retain_current:'Retain current labels',compare_proposal:'Compare labels',check_source_text:'Check source text'};
const findingNames={aspect_comparison:'Service aspect comparison',candidate_evidence_incomplete:'Source excerpt check',case_variant_consistency:'Capitalisation consistency',empty_source_text:'Source text',full_text_recheck:'Full-text check',geographic_context:'Location context',interpretation_uncertain:'Interpretation',language_context:'Language context',mixed_quote_support:'Mixed sentiment support',nli_sentiment_check:'Sentiment cross-check',original_polarity_coherence:'Earlier sentiment consistency',original_quote_check:'Earlier source excerpts',priority_comparison:'Priority comparison',punctuation_variant_consistency:'Punctuation consistency',quote_polarity_check:'Excerpt sentiment',rating_context:'Rating context',redacted_context:'Available context',referral_comparison:'Referral comparison',sentiment_comparison:'Sentiment comparison',shared_family_outside_scope:'Related feedback'};
const value=s=>s?e(title(s)):'—';
const aspectList=aspects=>Object.entries(aspects||{}).map(([key,sentiment])=>'<span class="audit-aspect">'+e(aspectNames[key]||title(key))+' · '+value(sentiment)+'</span>').join('')||'—';
const teams=list=>list?.length?list.map(team=>e(title(team))).join(' · '):'—';

export function renderAudit(audit,blind=false){
 if(!audit)return '';
 const header='<section class="audit-panel" aria-label="Second-pass audit"><div class="audit-heading"><h3>Second-pass audit</h3><span class="audit-status">Awaiting review</span></div>';
 if(blind)return header+'</section>';
 const earlier=audit.earlier,candidate=audit.candidate;
 const comparisons=[['Sentiment',value(earlier.sentiment),value(candidate.sentiment)],['Priority',value(earlier.priority),value(candidate.priority)],['Service aspects',aspectList(earlier.aspects),aspectList(candidate.aspects)],['Referral',teams(earlier.referral),teams(candidate.referral)]];
 return header+'<p class="audit-recommendation"><span>Recommendation</span><strong>'+e(recommendationNames[audit.recommendation]||title(audit.recommendation))+'</strong></p>'+
 '<h4>Findings</h4>'+(audit.findings.length?'<ul class="audit-findings">'+audit.findings.map(f=>'<li>'+e(findingNames[f]||title(f))+'</li>').join('')+'</ul>':'<p class="audit-empty">No additional findings.</p>')+
 '<h4>Label comparison</h4><div class="audit-comparison-wrap"><table class="audit-comparison"><thead><tr><th>Field</th><th>Earlier labels</th><th>Candidate labels</th></tr></thead><tbody>'+comparisons.map(([field,old,current])=>'<tr><th>'+field+'</th><td>'+old+'</td><td>'+current+'</td></tr>').join('')+'</tbody></table></div>'+
 '<h4>Source evidence <span class="audit-evidence-count">'+audit.evidence.length+'</span></h4>'+(audit.evidence.length?'<ol class="audit-evidence">'+audit.evidence.map(quote=>'<li><div><strong>'+e(aspectNames[quote.aspect]||title(quote.aspect))+'</strong><span class="sentiment-chip '+e(quote.sentiment)+'">'+value(quote.sentiment)+'</span></div><blockquote>'+e(quote.quote)+'</blockquote></li>').join('')+'</ol>':'<p class="audit-empty">No recorded excerpts.</p>')+'</section>';
}
