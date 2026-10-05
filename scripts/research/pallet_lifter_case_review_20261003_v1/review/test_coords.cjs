'use strict';
const assert = require('node:assert/strict');
const coords = require('./coords.js');
let cases=0;
for(const dpr of [1,1.25,2,3]) {
  for(const scale of [0.25,1,2.75,12]) {
    for(const rect of [{left:37,top:84,width:800,height:600},{left:-20,top:7,width:400,height:180}]) {
      const view={scale,tx:-73.25,ty:41.75,viewportWidth:800,viewportHeight:600};
      for(const raw of [{x:0,y:0},{x:639.9,y:479.9},{x:113.125,y:230.875}]) {
        const screen=coords.originalToScreen(raw,rect,view);
        const roundtrip=coords.screenToOriginal(screen,rect,view);
        assert.ok(Math.abs(raw.x-roundtrip.x)<1e-10);
        assert.ok(Math.abs(raw.y-roundtrip.y)<1e-10);
        // A high DPI backing buffer has no effect on CSS/client coordinates.
        const backing={x:(screen.x-rect.left)*dpr,y:(screen.y-rect.top)*dpr};
        const client={x:rect.left+backing.x/dpr,y:rect.top+backing.y/dpr};
        const fromBacking=coords.screenToOriginal(client,rect,view);
        assert.ok(Math.abs(raw.x-fromBacking.x)<1e-10);
        const zoomed=coords.zoomAt(screen,rect,view,1.6);
        const fixed=coords.originalToScreen(raw,rect,zoomed);
        assert.ok(Math.abs(fixed.x-screen.x)<1e-10);
        assert.ok(Math.abs(fixed.y-screen.y)<1e-10);
        cases++;
      }
    }
  }
}
const rect={left:50,top:20,width:500,height:250};
const view={scale:2,tx:10,ty:-30,viewportWidth:1000,viewportHeight:500};
const point=coords.screenToOriginal({x:155,y:55},rect,view);
assert.deepEqual(point,{x:100,y:50});
console.log(JSON.stringify({passed:true,roundtrip_zoom_cases:cases,known_css_scaled_point:point}));
