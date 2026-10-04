'use strict';
// Scale from the original decoded image; retain its PNG, JPEG or WebP format.
// The native processor handles patch alignment after upload.
window.ClefImages = (() => {
  const maxBytes = 10 * 1024 * 1024;
  function dimensions(width, height, preset) {
    if (![width, height].every(value => Number.isFinite(value) && value > 0)) {
      throw new Error('Invalid image dimensions.');
    }
    if (Math.max(width, height) / Math.min(width, height) > 200) {
      throw new Error('The image is too narrow or wide for the vision processor.');
    }
    let scale = 1;
    if (!preset.original) {
      let {width: limitWidth, height: limitHeight} = preset;
      if (![limitWidth, limitHeight].every(value => Number.isFinite(value) && value > 0)) {
        throw new Error('Invalid scaling dimensions.');
      }
      if (height > width && limitWidth > limitHeight) [limitWidth, limitHeight] = [limitHeight, limitWidth];
      scale = Math.min(1, limitWidth / width, limitHeight / height);
    }
    return {width: Math.max(1, Math.floor(width * scale)), height: Math.max(1, Math.floor(height * scale))};
  }
  async function prepare(item, preset) {
    const key = JSON.stringify(preset);
    if (item.prepared?.key === key) return item.prepared;
    const sourceWidth = item.image.naturalWidth, sourceHeight = item.image.naturalHeight;
    const {width, height} = dimensions(sourceWidth, sourceHeight, preset);
    const format = item.format;
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(format)) throw new Error(`${item.name}: unsupported image format.`);
    // Original and already-small inputs keep their exact uploaded file bytes.
    if (width === sourceWidth && height === sourceHeight) {
      item.prepared = {key, width, height, data: item.data, bytes: item.bytes, format};
      return item.prepared;
    }
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('This browser cannot prepare images.');
    context.imageSmoothingEnabled = true; context.imageSmoothingQuality = 'high';
    context.drawImage(item.image, 0, 0, width, height);
    const blob = await new Promise((resolve, reject) => canvas.toBlob(value => {
      if (value) resolve(value); else reject(new Error(`Could not prepare ${item.name}.`));
    }, format, 0.95));
    // Unsupported canvas encoders can silently return PNG. Never change format.
    if (blob.type !== format) throw new Error(`${item.name}: this browser cannot resize ${format.split('/')[1].toUpperCase()} while preserving its format. Use Original or another browser.`);
    if (blob.size > maxBytes) throw new Error(`${item.name}: scaled image exceeds 10 MB. Choose a smaller scaling option.`);
    const data = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error(`Could not prepare ${item.name}.`));
      reader.readAsDataURL(blob);
    });
    item.prepared = {key, width, height, data, bytes: blob.size, format};
    return item.prepared;
  }
  return {dimensions, prepare};
})();
