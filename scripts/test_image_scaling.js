// CPU checks for client scaling geometry; run with node scripts/test_image_scaling.js.
'use strict';
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const path = require('node:path');
const scope = {window: {}};
vm.createContext(scope);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/image-input.js'), 'utf8'), scope);
const dimensions = scope.window.ClefImages.dimensions;
const presets = [[256, 256], [320, 200], [640, 480], [1280, 720], [1920, 1080], [2560, 1440], [3840, 2160]];
let count = 0;
for (const [w, h] of [[3024, 4032], [4032, 3024], [1786, 2099], [64, 64], [32, 33], [1920, 1080], [8192, 1024]]) {
  for (const [width, height] of presets) {
    const d = dimensions(w, h, {width, height});
    const [bw, bh] = w < h && width > height ? [height, width] : [width, height];
    assert(d.width <= bw && d.height <= bh, 'Image must fit the selected frame');
    assert(d.width <= w && d.height <= h, 'Content must not be enlarged');
    const roundingTolerance = Math.max(w / h, 1) / Math.min(d.width, d.height);
    assert(Math.abs(d.width / d.height - w / h) < roundingTolerance, 'Preserve content aspect ratio within integer rounding');
    count++;
  }
  const original = dimensions(w, h, {original: true});
  assert.equal(original.width, w); assert.equal(original.height, h);
  count++;
}
assert.throws(() => dimensions(1000, 1, {original: true}));
assert.throws(() => dimensions(0, 1, {original: true}));
console.log(`PASS: ${count} source/preset combinations, invalid dimensions and unsupported aspect ratio`);

// Guard against browsers silently substituting PNG for an unsupported encoder.
(async () => {
  let encodedFormat = 'image/png', encodedBytes = 100, encodes = 0;
  scope.document = {createElement() {return {
    getContext() {return {drawImage() {}};},
    toBlob(callback, format) {encodes++; callback({type: encodedFormat, size: encodedBytes});}
  };}};
  scope.FileReader = class {
    readAsDataURL(blob) {this.result = `data:${blob.type};base64,fixture`; this.onload();}
  };
  const prepare = scope.window.ClefImages.prepare;
  const item = {name: 'fixture.webp', format: 'image/webp', bytes: 50,
    data: 'data:image/webp;base64,original', image: {naturalWidth: 800, naturalHeight: 600}};
  const original = await prepare(item, {original: true});
  assert.equal(original.data, item.data); assert.equal(encodes, 0);
  await assert.rejects(() => prepare(item, {width: 256, height: 256}), /cannot resize WEBP/);
  encodedFormat = 'image/webp';
  const scaled = await prepare(item, {width: 256, height: 256});
  assert.equal(scaled.format, item.format);
  const before = encodes;
  assert.equal(await prepare(item, {width: 256, height: 256}), scaled);
  assert.equal(encodes, before, 'Repeat requests must reuse prepared bytes');
  encodedBytes = 11 * 1024 * 1024;
  await assert.rejects(() => prepare(item, {width: 128, height: 128}), /exceeds 10 MB/);
  assert.equal(encodes, before + 1, 'Oversized outputs must not try a different format');
  console.log('PASS: original bypass, preserved format, repeat reuse, encoder substitution and oversize rejection');
})().catch(error => {console.error(error); process.exitCode = 1;});
