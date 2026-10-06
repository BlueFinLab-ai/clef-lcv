'use strict';
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(require('node:path').join(__dirname, '../web/app.js'), 'utf8');
function body(name, next) {return source.slice(source.indexOf(`function ${name}(`), source.indexOf(`function ${next}(`));}
const context = vm.createContext({location: {origin: 'https://private-host:9999'}, demoMode: true, apiKeyRequired: true,
  apiKey: 'actual-secret-must-not-appear', images: [], pythonCode: Symbol(),
  exampleRequest: () => ({state: 'test', questions: {test: {type: 'noul'}}})});
vm.runInContext(body('apiBase', 'exampleRequest') + body('pythonLiteral', 'curlExample') + body('curlExample', 'pythonExample') +
  source.slice(source.indexOf('function pythonExample('), source.indexOf('const codeExamples')), context);
for (const mode of [true, false]) for (const auth of [true, false]) {
  context.demoMode = mode; context.apiKeyRequired = auth;
  for (const fn of ['curlExample()', 'pythonExample()']) {
    const text = vm.runInContext(fn, context);
    assert.equal(text.includes('YOUR_API_KEY'), auth);
    assert.equal(text.includes('http://<HOST_NAME>:<PORT>'), mode);
    assert.equal(text.includes('private-host'), !mode);
    assert.ok(!text.includes(context.apiKey));
  }
}
console.log('PASS: URL masking toggles, quoted curl placeholders, Bearer examples, and no entered key in generated code');
