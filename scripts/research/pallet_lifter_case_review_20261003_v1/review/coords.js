'use strict';
(function (root) {
  function screenToOriginal(point, rect, view) {
    return {
      x: ((point.x - rect.left) * view.viewportWidth / rect.width - view.tx) / view.scale,
      y: ((point.y - rect.top) * view.viewportHeight / rect.height - view.ty) / view.scale
    };
  }
  function originalToScreen(point, rect, view) {
    return {
      x: rect.left + (point.x * view.scale + view.tx) * rect.width / view.viewportWidth,
      y: rect.top + (point.y * view.scale + view.ty) * rect.height / view.viewportHeight
    };
  }
  function zoomAt(point, rect, view, factor) {
    const original = screenToOriginal(point, rect, view);
    const next = Object.assign({}, view, {scale: Math.min(30, Math.max(0.1, view.scale * factor))});
    next.tx = (point.x - rect.left) * next.viewportWidth / rect.width - original.x * next.scale;
    next.ty = (point.y - rect.top) * next.viewportHeight / rect.height - original.y * next.scale;
    return next;
  }
  const api = {screenToOriginal, originalToScreen, zoomAt};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.ReviewCoords = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
