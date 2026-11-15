import fs from 'node:fs';import crypto from 'node:crypto';
const manifest=JSON.parse(fs.readFileSync('public/data-app-build.json'));
for(const key of ['html','snapshot']){const item=manifest[key],bytes=fs.readFileSync('public/'+item.path);if(crypto.createHash('sha256').update(bytes).digest('hex')!==item.sha256)throw Error('Release integrity mismatch: '+key);}
const data=JSON.parse(fs.readFileSync('public/'+manifest.snapshot.path));if(data.buildStatus!=='complete')throw Error('Dashboard is unfinished');console.log('Verified static dashboard and trained-model snapshot.');
