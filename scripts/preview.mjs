import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const types={'.html':'text/html;charset=utf-8','.js':'text/javascript;charset=utf-8','.css':'text/css;charset=utf-8','.json':'application/json','.gz':'application/gzip','.ttf':'font/ttf','.svg':'image/svg+xml'};
for(const [port,variant] of [[8123,'Original'],[8124,'UI-fixed']]){http.createServer(async(req,res)=>{try{const url=new URL(req.url,'http://127.0.0.1');let name=decodeURIComponent(url.pathname);if(name==='/'){res.writeHead(302,{Location:'/'+variant+'/'});res.end();return;}if(name.endsWith('/'))name+='index.html';const target=path.resolve(root,'.'+name);if(!target.startsWith(root+path.sep)||name.split('/').some(v=>v.startsWith('.')||/reviewer/i.test(v))){res.writeHead(404);res.end();return;}const content=await fs.readFile(target);res.writeHead(200,{'Content-Type':types[path.extname(target)]||'application/octet-stream','Cache-Control':'no-store'});res.end(content);}catch{res.writeHead(404);res.end('Not found');}}).listen(port,'127.0.0.1',()=>console.log('Preview http://127.0.0.1:'+port+'/'+variant+'/'));}
