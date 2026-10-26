// Local DOM-shim unit checks only. No browser, network, UI control, or image execution.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const W=__dirname,source=fs.readFileSync(path.join(W,'ROOT_READOUT_KO.html'),'utf8');
const matches=[...source.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)];assert.equal(matches.length,2);
const data=JSON.parse(matches[0][2]);const ids=[...source.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]);assert.equal(new Set(ids).size,ids.length);
class Element{constructor(id){this.id=id;this.value='';this.checked=false;this.children=[];this.events={};this._html='';this.dataset={};this.textContent=''}
 appendChild(e){this.children.push(e);if(this.children.length===1)this.value=e.value}
 addEventListener(n,f){this.events[n]=f}
 set innerHTML(s){this._html=s}get innerHTML(){return this._html}}
const elements=Object.fromEntries(ids.map(x=>[x,new Element(x)]));elements['saved-data'].textContent=matches[0][2];
Object.assign(elements.mode,{value:'cutout_png'});elements.family.value='historical';elements.kp.value='all';elements.group.value='ALL';elements['metric-slot'].value='ALL';
const document={getElementById:id=>{assert(elements[id],id);return elements[id]},createElement:tag=>new Element(tag),querySelectorAll:selector=>{assert.equal(selector,'[data-kp]');return [...elements['point-table'].innerHTML.matchAll(/data-kp="(\d+)"/g)].map(x=>{let e=new Element('kp');e.dataset.kp=x[1];return e})}};
vm.runInNewContext(matches[1][2],{document,console,Number,String,Math,JSON},{timeout:2000});
const checks=[];function check(name,f){f();checks.push(name)}
function assets(){return [...elements.images.innerHTML.matchAll(/src="([^"]+)"/g)].map(x=>x[1])}
function verifyAssets(){assert.equal(assets().length,4);for(const rel of assets()){assert(!path.isAbsolute(rel)&&!rel.split('/').includes('..'));assert(fs.statSync(path.join(W,rel)).isFile())}}
check('initial10cases_and4actual_images',()=>{assert.equal(elements.scene.children.length,10);verifyAssets();assert(assets().slice(1).every(x=>x.endsWith('_CUTOUT.png')))});
check('binary_masks_use_actual_binary_pngs',()=>{elements.mode.value='binary_png';elements.mode.events.change();verifyAssets();assert(assets().slice(1).every(x=>x.endsWith('_BINARY.png')))});
check('same_backbone_EVA_group',()=>{elements.family.value='matched';elements.family.events.change();assert(assets()[1].includes('EVA_STANDARD'));assert(assets()[2].includes('EVA_CSA'));assert(assets()[3].includes('EVA_DECLIP'))});
check('all10frames5models4display_modes_files_exist',()=>{for(let i=0;i<10;i++){elements.scene.value=String(i);for(const family of ['historical','matched'])for(const mode of ['binary_png','cutout_png','overlay_png','probability_png']){elements.family.value=family;elements.mode.value=mode;elements.scene.events.change();verifyAssets()}}});
check('query_and_manual_reference_markers_render',()=>{elements.scene.value='0';elements.kp.value='2';elements['show-query'].checked=true;elements['show-manual'].checked=true;elements.kp.events.change();assert(elements.images.innerHTML.includes('<circle'));assert(elements['point-table'].innerHTML.includes('focus'))});
check('next_and_previous_scene_update',()=>{elements.scene.value='0';elements.next.onclick();assert.equal(Number(elements.scene.value),1);elements.previous.onclick();assert.equal(Number(elements.scene.value),0)});
check('all3materials_all8native_filters',()=>{for(const g of ['ALL','WOOD_PRIMARY105','SQUARE118'])for(const slot of ['ALL','0','1','2','3','4','5','6','7']){elements.group.value=g;elements['metric-slot'].value=slot;elements.group.events.change();assert.equal((elements['metric-table'].innerHTML.match(/<tr>/g)||[]).length,11)}});
const out={PASS:true,checks,execution:'Node VM and local DOM shim; not a browser render or screenshot test',network_calls:0,browser_UI_actions:0};fs.writeFileSync(path.join(W,'ROOT_JS_INTERACTION_UNIT_QA.json'),JSON.stringify(out,null,2)+'\n');console.log(JSON.stringify(out));
