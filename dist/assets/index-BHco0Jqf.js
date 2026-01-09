import{aO as E,e0 as M,a as O,ei as Bt,eZ as on,dZ as zt,cT as ut,bO as li,a6 as z,e2 as de,e_ as ui,eh as Ht,B as di,e$ as pi,bP as $e,e6 as Xt,f0 as hi,e8 as Zs,dp as fe,b5 as Xe,e4 as $t,e5 as bt,z as Ke,ah as me,da as se,f1 as ze,bT as fi,bR as mi,e1 as oe,d3 as xi,dg as $n,f2 as gi,bN as Ci,U as $i,bS as bi,e7 as ye,bx as vi,f3 as wi,dX as vt,d1 as Js,bK as ge,eg as Kt,ea as dt,eT as eo,e3 as He,dh as to,bb as no,eb as Ii,d7 as so,f4 as oo,d2 as bn,as as j,N as vn,e9 as B,P as wn,a7 as In,ao as Rn,aq as yn,at as Sn,df as Tn,av as En,aw as Nn,aB as kn,aC as An,aG as On,dl as Dn,aY as Fn,dq as Pn,ed as je,aI as ro,b0 as _n,bF as ao,b6 as io,ag as X,t as co,r as Ri,a8 as yi,bf as Ln,bh as rn,dz as Vn,S as lo,dA as Bn,bp as Mn,cg as Wn,f5 as Si,dD as Un,f6 as ot,ck as w,f7 as an,f8 as Cs,f9 as Gn,bq as Pe,i as q,fa as uo,fb as $s,dU as Ti,dV as Ei,dW as we,dI as Ee,cV as bs,fc as Ni,bD as ki,cS as Ai,am as Oi,fd as Di,fe as Fi,d_ as Pi,di as _i,ds as Li,dw as Vi,d$ as Bi,aE as Mi,es as zn,dE as Wi,bX as Ui,A as Gi,h as zi,j as Hi,k as Xi,l as Ki,n as ji,p as qi,q as Yi,u as Qi,v as Zi,x as Ji,y as ec,C as tc,D as nc,d4 as sc,d5 as oc,I as rc,L as ac,K as ic,M as cc,O as lc,aV as uc,R as dc,d8 as pc,ax as hc,H as fc,V as mc,bM as xc,W as gc,X as Cc,d9 as $c,Y as bc,Z as vc,_ as wc,bY as Ic,$ as Rc,a0 as yc,a1 as Sc,a2 as Tc,a3 as Ec,bU as Nc,bV as kc,a4 as Ac,a5 as Oc,ab as Dc,dd as Fc,de as Pc,ad as _c,ap as Lc,bm as Vc,ee as Bc,ef as Mc,bZ as Wc,cF as Uc,bQ as Gc,bW as zc,bJ as Hc,au as Xc,bn as Kc,ay as jc,az as qc,aA as Yc,aD as Qc,aH as Zc,aL as Jc,aM as el,aN as tl,aF as nl,dk as sl,ae as ol,aQ as rl,aR as al,dm as il,dn as cl,aS as ll,aT as ul,af as dl,aZ as pl,a_ as hl,dt as fl,bl as ml,a$ as xl,b$ as gl,c0 as Cl,c3 as $l,c4 as bl,c1 as vl,c2 as wl,b1 as Il,dF as Rl,b2 as yl,br as Sl,bE as Tl,b3 as El,dr as Nl,b7 as kl,b8 as Al,b9 as Ol,ba as Dl,bc as Fl,dv as Pl,du as _l,c5 as Ll,dx as Vl,c6 as Bl,dy as Ml,bd as Wl,b_ as Ul,be as Gl,bH as zl,aP as Hl,a9 as Xl,bg as Kl,bi as jl,bj as ql,bk as Yl,aJ as Ql,b4 as Zl,c9 as Jl,ca as eu,cb as tu,cc as nu,bI as su,bo as ou,dB as ru,dC as au,bs as iu,cd as cu,ce as lu,cf as uu,bt as du,J as pu,by as hu,ar as fu,bz as mu,c7 as xu,bA as gu,bC as Cu,bB as $u,ej as bu}from"./index-D3B-3C1D.js";function Hn(n,e){const t=n.shape.length,s=e.shape.length;if(t<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${t}.`);if(s<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${s}.`);if(e.dtype!=="int32")throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.shape[s-1]>t)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${e.shape[s-1]} vs. ${t}`);if(E(n.shape)===0)throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${n.shape}.`);const o=e.shape,r=o[o.length-1];let a=1;for(let d=0;d<o.length-1;++d)a*=o[d];const c=n.shape,i=o.slice();i.pop();let l=1;for(let d=r;d<t;++d)l*=c[d],i.push(c[d]);const u=[...M(n.shape).map(d=>d/l),1].slice(0,r);return[i,a,l,u]}const iv=Object.freeze(Object.defineProperty({__proto__:null,prepareAndValidate:Hn},Symbol.toStringTag,{value:"Module"}));const cn=-2,vu=-1;function Xn(n,e,t){const s=n.shape.length;O(s===e.length,()=>`Error in slice${s}D: Length of begin ${e} must match the rank of the array (${s}).`),O(s===t.length,()=>`Error in slice${s}D: Length of size ${t} must match the rank of the array (${s}).`);for(let o=0;o<s;++o)O(e[o]+t[o]<=n.shape[o],()=>`Error in slice${s}D: begin[${o}] + size[${o}] (${e[o]+t[o]}) would overflow input.shape[${o}] (${n.shape[o]})`)}function wu(n){const e=[];let t=0;for(;n>0;)n&1&&e.push(t),n/=2,t++;return e}function po(n,e,t){const s=[];for(let o=0;o<n.length;o++)s[o]=Math.ceil((e[o]-n[o])/t[o]);return s}function ho(n,e,t,s){const o=[...n];for(let r=o.length;r<s.length;r++)o.push(1);for(let r=0;r<t;r++)r===0?o[e]=1:(o.splice(e,0,1),o.pop());return o}function fo(n,e,t){return t<=n?t:t-(e-1)}function mo(n,e){const t=[];for(let s=0;s<n;s++)t.push(e+s);return t}function Iu(n,e,t,s,o,r,a,c,i){const l=n.length;let u=new Array(l),d=new Array(l),p=new Array(l);if(e.length&&t>0){const h=e[0],f=t+1;u=xo(a,h,f,s,n),d=go(c,h,f,o,n),p=ho(r,h,f,n)}else for(let h=0;h<l;h++)u[h]=$o(a,s,r,n,h,i),d[h]=bo(c,o,r,n,h,i),p[h]=Co(r,h,i);return{begin:u,end:d,strides:p}}function xo(n,e,t,s,o){const r=[...o],a=mo(t,e);for(let c=0;c<r.length;c++)if(a.indexOf(c)>-1)r[c]=0;else{const i=fo(e,t,c);let l=s[i];n&1<<i&&(l=0),r[c]=l}return r}function go(n,e,t,s,o){const r=[...o],a=mo(t,e);for(let c=0;c<r.length;c++)if(a.indexOf(c)>-1)r[c]=Number.MAX_SAFE_INTEGER;else{const i=fo(e,t,c);let l=s[i];n&1<<i&&(l=Number.MAX_SAFE_INTEGER),r[c]=l}for(let c=0;c<r.length;c++){const i=o[c];r[c]<0&&(r[c]+=i),r[c]=Bt(0,r[c],o[c])}return r}function Co(n,e,t){let s=n[e];return(t&1<<e||s==null)&&(s=1),s}function $o(n,e,t,s,o,r){let a=e[o];const c=t[o]||1;(n&1<<o||r&1<<o||a==null)&&(c>0?a=Number.MIN_SAFE_INTEGER:a=Number.MAX_SAFE_INTEGER);const i=s[o];return a<0&&(a+=i),a=Bt(0,a,i-1),a}function bo(n,e,t,s,o,r){let a=e[o];const c=t[o]||1;(n&1<<o||r&1<<o||a==null)&&(c>0?a=Number.MAX_SAFE_INTEGER:a=Number.MIN_SAFE_INTEGER);const i=s[o];return a<0&&(a+=i),c>0?a=Bt(0,a,i):a=Bt(-1,a,i-1),a}function Kn(n,e,t){let s=t.length;for(let o=0;o<t.length;o++)if(t[o]>1){s=o;break}for(let o=s+1;o<t.length;o++)if(e[o]>0||t[o]!==n[o])return!1;return!0}function jn(n,e){let t=n.length>0?n[n.length-1]:1;for(let s=0;s<n.length-1;s++)t+=n[s]*e[s];return t}function qn(n,e,t){let s;const o=n.shape.length;typeof e=="number"?s=[e,...new Array(o-1).fill(0)]:e.length<o?s=e.concat(new Array(o-e.length).fill(0)):s=e.slice(),s.forEach(a=>{O(a!==-1,()=>"slice() does not support negative begin indexing.")});let r;return t==null?r=new Array(o).fill(-1):typeof t=="number"?r=[t,...new Array(o-1).fill(-1)]:t.length<o?r=t.concat(new Array(o-t.length).fill(-1)):r=t,r=r.map((a,c)=>a>=0?a:(O(a===-1,()=>`Negative size values should be exactly -1 but got ${a} for the slice() size at index ${c}.`),n.shape[c]-s[c])),[s,r]}function vo(n,e,t,s,o,r,a,c,i){let l;if(s==null?(l=new Array(e.length),l.fill(1)):l=s,a!=null&&(a&a-1)!==0)throw new Error("Multiple ellipses in slice is not allowed.");let u=!1;const d={dims:l.length,numAddAxisAfterEllipsis:0,begin:e.slice(),end:t.slice(),strides:l.slice(),beginMask:o,endMask:r,ellipsisMask:a,newAxisMask:c,shrinkAxisMask:i};for(let $=0;$<d.dims;$++)u&&(1<<$&c)!==0&&d.numAddAxisAfterEllipsis++,1<<$&a&&(u=!0);u||(d.ellipsisMask|=1<<d.dims,d.dims++);const p={dims:n.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};Ru(d,p);let h=!0,f=!0,g=!0;const x=[],m=[];for(let $=0;$<n.length;++$){if(p.strides[$]===0)throw Error(`strides[${$}] must be non-zero`);const b=!!(p.shrinkAxisMask&1<<$),v=n[$];if(v===-1){x.push(b?1:-1);continue}const T=[p.beginMask&1<<$,p.endMask&1<<$],S=[p.strides[$]>0?0:-1,p.strides[$]>0?v:v-1];if(b&&p.strides[$]<=0)throw Error("only stride 1 allowed on non-range indexing.");g=g&&p.strides[$]===1;const I=!!(p.beginMask&1<<$&&p.endMask&1<<$);if(p.beginValid&&p.endValid){if(b){const P=p.begin[$]<0?v+p.begin[$]:p.begin[$];if(p.begin[$]=P,p.end[$]=p.begin[$]+1,P<0||P>=v)throw Error(`slice index ${p.begin[$]} of dimension ${$} out of bounds.`)}else p.begin[$]=vs(p.begin[$],0,p.strides[$],v,T,S),p.end[$]=vs(p.end[$],1,p.strides[$],v,T,S);const F=p.strides[$]===1&&p.begin[$]===0&&p.end[$]===v;h=h&&F,f=f&&($===0&&p.strides[$]===1||F)}else h=h&&p.strides[$]===1&&I,f=f&&($===0&&p.strides[$]===1||I);let A,k=!1;if(p.beginValid&&p.endValid?(A=p.end[$]-p.begin[$],k=!0):b?(A=1,k=!0):I&&v>=0&&(p.strides[$]<0?A=-v:A=v,k=!0),k){let F;A===0||A<0!=p.strides[$]<0?F=0:F=Math.trunc(A/p.strides[$])+(A%p.strides[$]!==0?1:0),x.push(F)}else x.push(-1)}for(let $=0;$<p.finalShapeGatherIndices.length;++$){const b=p.finalShapeGatherIndices[$];b>=0?m.push(x[b]):b===cn&&m.push(1)}return{finalShapeSparse:m.filter(($,b)=>p.finalShapeGatherIndices[b]!==cn),finalShape:m,isIdentity:h,sliceDim0:f,isSimpleSlice:g,begin:p.begin,end:p.end,strides:p.strides}}function Ru(n,e){e.beginMask=0,e.endMask=0,e.shrinkAxisMask=0;let t=0;e.beginValid=n.begin!=null,e.endValid=n.end!=null,e.begin=new Array(e.dims),e.end=new Array(e.dims),e.strides=new Array(e.dims),e.finalShapeGatherIndices=[],e.finalShapeGatherIndicesSparse=[],e.inputShapeGatherIndicesSparse=new Array(e.dims);for(let s=0;s<n.dims;s++)if(1<<s&n.ellipsisMask){const o=Math.min(e.dims-(n.dims-s)+1+n.numAddAxisAfterEllipsis,e.dims);for(;t<o;t++)e.begin[t]=0,e.end[t]=0,e.strides[t]=1,e.beginMask|=1<<t,e.endMask|=1<<t,e.finalShapeGatherIndices.push(t),e.finalShapeGatherIndicesSparse.push(-1),e.inputShapeGatherIndicesSparse[t]=s}else if(1<<s&n.newAxisMask)e.finalShapeGatherIndices.push(cn),e.finalShapeGatherIndicesSparse.push(-1);else{if(t===e.begin.length)throw Error(`Index out of range using input dim ${t}; input has only ${e.dims} dims, ${e.begin.length}.`);n.begin!=null&&(e.begin[t]=n.begin[s]),n.end!=null&&(e.end[t]=n.end[s]),e.strides[t]=n.strides[s],n.beginMask&1<<s&&(e.beginMask|=1<<t),n.endMask&1<<s&&(e.endMask|=1<<t),n.shrinkAxisMask&1<<s?(e.finalShapeGatherIndices.push(vu),e.finalShapeGatherIndicesSparse.push(-1),e.shrinkAxisMask|=1<<t):(e.finalShapeGatherIndices.push(t),e.finalShapeGatherIndicesSparse.push(s)),e.inputShapeGatherIndicesSparse[t]=s,t++}}function vs(n,e,t,s,o,r){if(o[e])return t>0?r[e]:r[e+1&1];{const a=n<0?s+n:n;return a<r[0]?r[0]:a>r[1]?r[1]:a}}const yu=Object.freeze(Object.defineProperty({__proto__:null,assertParamsValid:Xn,computeFlatOffset:jn,computeOutShape:po,getNormalizedAxes:Iu,isSliceContinous:Kn,maskToAxes:wu,parseSliceParams:qn,sliceInfo:vo,startForAxis:$o,startIndicesWithElidedDims:xo,stopForAxis:bo,stopIndicesWithElidedDims:go,stridesForAxis:Co,stridesWithElidedDims:ho},Symbol.toStringTag,{value:"Module"}));const Su=typeof requestAnimationFrame<"u"?requestAnimationFrame:typeof setImmediate<"u"?setImmediate:n=>n();function Tu(){return new Promise(n=>Su(()=>n()))}function wo(n,e){const t=n[0].length;n.forEach((o,r)=>{O(o.length===t,()=>`Error in concat${t}D: rank of tensors[${r}] must be the same as the rank of the rest (${t})`)}),O(e>=0&&e<t,()=>`Error in concat${t}D: axis must be between 0 and ${t-1}.`);const s=n[0];n.forEach((o,r)=>{for(let a=0;a<t;a++)O(a===e||o[a]===s[a],()=>`Error in concat${t}D: Shape of tensors[${r}] (${o}) does not match the shape of the rest (${s}) along the non-concatenated axis ${r}.`)})}function Ae(n,e){const t=n[0].slice();for(let s=1;s<n.length;s++)t[e]+=n[s][e];return t}var le;(function(n){n[n.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",n[n.VALUE_ROWIDS=1]="VALUE_ROWIDS",n[n.ROW_LENGTHS=2]="ROW_LENGTHS",n[n.ROW_SPLITS=3]="ROW_SPLITS",n[n.ROW_LIMITS=4]="ROW_LIMITS",n[n.ROW_STARTS=5]="ROW_STARTS"})(le||(le={}));function Io(n,e,t){let s=new Array;if(t==null&&e==null)return s;if(e==null)for(;s.length<n+t.length;)s.push(-1);else s=e.slice();if(t==null)return s;if(n+t.length!==s.length)throw new Error(`rt input.shape and shape=${e} are incompatible: rt input.rank = ${n+t.length}, but shape.rank = ${s.length}`);for(let o=1;o<t.length;++o){const r=t[o],a=s[s.length-t.length+o],c=s[a];if(r>=0)if(c>=0){if(c!==r)throw new Error(`rt input.shape and shape=${e} are incompatible: rt input.shape[${o+n}] = ${r} but shape[${o+n}] = ${c}`)}else s[a]=r}return s}function Ro(n){const e={FIRST_DIM_SIZE:le.FIRST_DIM_SIZE,VALUE_ROWIDS:le.VALUE_ROWIDS,ROW_LENGTHS:le.ROW_LENGTHS,ROW_SPLITS:le.ROW_SPLITS,ROW_LIMITS:le.ROW_LIMITS,ROW_STARTS:le.ROW_STARTS},t=[];for(const s of n)if(s in e)t.push(e[s]);else break;return t}function yo(n){return n.length===0?0:n[0]===le.FIRST_DIM_SIZE?n.length-1:n.length}function So(n,e){if(n==null||e==null)return;const t=n.length,s=e.length;if(t>=s)throw new Error(`defaultValue.shape=${n} and ragged tensor flatValues.shape=${e}, are incompatible: defaultValue.rank = ${t} must be less than ragged tensor input flatValues.rank = ${s})`);for(let o=0;o<Math.min(t,s-1);++o){const r=n[o],a=e[o+1];if(r>=0&&a>=0&&r!==1&&r!==a)throw new Error(`defaultValue.shape=${n}, and ragged tensor input flatValues.shape=${e} are incompatible: defaultValue.shape[${o-n.length}] = ${r} but ragged tensor input.flatValues.shape[${o-n.length}] = ${a}`)}}const Yn=30;function jt(n){return n<=Yn?n:on(n,Math.floor(Math.sqrt(n)))}function To(n,e,t){const s=t*(typeof n=="number"?n:n[0]),o=e*(typeof n=="number"?n:n[1]);return[s,o]}function Qn(n,e,t,s=!0){let o=[];if(s)o=o.concat(e.slice(0)),o.push(n[0]/t),o=o.concat(n.slice(1));else{o=o.concat(n[0]);const r=e.length;for(let a=0;a<r;++a)o=o.concat([n[a+1]/e[a],e[a]]);o=o.concat(n.slice(r+1))}return o}function Zn(n,e,t=!0){const s=[];if(t){s.push(e);for(let o=e+1;o<n;++o)o<=2*e?(s.push(o),s.push(o-(e+1))):s.push(o)}else{const o=[],r=[];for(let a=1;a<n;++a)a>=e*2+1||a%2===1?r.push(a):o.push(a);s.push(...o),s.push(0),s.push(...r)}return s}function Jn(n,e,t,s=!0){const o=[];s?o.push(n[0]/t):o.push(n[0]*t);for(let r=1;r<n.length;++r)r<=e.length?s?o.push(e[r-1]*n[r]):o.push(n[r]/e[r-1]):o.push(n[r]);return o}function Eo(n,e){const t=[0];for(let s=0;s<e;++s)t.push(n[s][0]);return t}function No(n,e,t){const s=n.slice(0,1);for(let o=0;o<t;++o)s.push(n[o+1]-e[o][0]-e[o][1]);return s}const ko=1.7580993408473768,Ao=1.0507009873554805;const Oo=.3275911,Do=.254829592,Fo=-.284496736,Po=1.421413741,_o=-1.453152027,Lo=1.061405429;function pt(n,e){if(n.length!==e.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${n.length}, imag: ${e.length}.`);const t=new Float32Array(n.length*2);for(let s=0;s<t.length;s+=2)t[s]=n[s/2],t[s+1]=e[s/2];return t}function Eu(n){const e=new Float32Array(n.length/2),t=new Float32Array(n.length/2);for(let s=0;s<n.length;s+=2)e[s/2]=n[s],t[s/2]=n[s+1];return{real:e,imag:t}}function Nu(n){const e=Math.ceil(n.length/4),t=new Float32Array(e),s=new Float32Array(e);for(let o=0;o<n.length;o+=4)t[Math.floor(o/4)]=n[o],s[Math.floor(o/4)]=n[o+1];return{real:t,imag:s}}function ku(n){const e=Math.floor(n.length/4),t=new Float32Array(e),s=new Float32Array(e);for(let o=2;o<n.length;o+=4)t[Math.floor(o/4)]=n[o],s[Math.floor(o/4)]=n[o+1];return{real:t,imag:s}}function Au(n,e){const t=n[e*2],s=n[e*2+1];return{real:t,imag:s}}function Ou(n,e,t,s){n[s*2]=e,n[s*2+1]=t}function Du(n,e){const t=new Float32Array(n/2),s=new Float32Array(n/2);for(let o=0;o<Math.ceil(n/2);o++){const r=(e?2:-2)*Math.PI*(o/n);t[o]=Math.cos(r),s[o]=Math.sin(r)}return{real:t,imag:s}}function Fu(n,e,t){const s=(t?2:-2)*Math.PI*(n/e),o=Math.cos(s),r=Math.sin(s);return{real:o,imag:r}}const nn="->",Pu=/->/g,ws=",",Is="...";function Vo(n,e){n=n.replace(/\s/g,"");const t=(n.length-n.replace(Pu,"").length)/nn.length;if(t<1)throw new Error("Equations without an arrow are not supported.");if(t>1)throw new Error(`Equation must contain exactly one arrow ("${nn}").`);const[s,o]=n.split(nn);O(s.indexOf(Is)===-1,()=>`The ellipsis notation ("${Is}") is not supported yet.`);const r=s.split(ws),a=r.length;if(e!==a)throw new Error(`Expected ${a} input tensors, received ${e}`);if(a>2)throw new Error("Support for more than 2 input tensors is not implemented yet.");const c=[];for(let p=0;p<o.length;++p){const h=o[p];if(!r.some(f=>f.indexOf(h)!==-1))throw new Error(`Output subscripts contain the label ${h} not present in the input subscripts.`);c.indexOf(h)===-1&&c.push(h)}for(let p=0;p<s.length;++p){const h=s[p];c.indexOf(h)===-1&&h!==ws&&c.push(h)}const i=new Array(r.length);for(let p=0;p<a;++p){if(new Set(r[p].split("")).size!==r[p].length)throw new Error(`Found duplicate axes in input component ${r[p]}. Support for duplicate axes in input is not implemented yet.`);i[p]=[];for(let h=0;h<r[p].length;++h)i[p].push(c.indexOf(r[p][h]))}const l=c.length,u=o.length,d=[];for(let p=u;p<l;++p)d.push(p);return{allDims:c,summedDims:d,idDims:i}}function Bo(n,e){let t=new Array(n);t.fill(-1);for(let o=0;o<e.length;++o)t[e[o]]=o;const s=[];for(let o=0;o<n;++o)t[o]===-1&&s.push(o);return t=t.filter(o=>o!==-1),{permutationIndices:t,expandDims:s}}function Mo(n,e,t){const s=new Array(n);for(let o=0;o<t.length;++o){const r=t[o].shape;for(let a=0;a<e[o].length;++a)s[e[o][a]]===void 0?s[e[o][a]]=r[a]:O(s[e[o][a]]===r[a],()=>`Expected dimension ${s[e[o][a]]} at axis ${a} of input shaped ${JSON.stringify(r)}, but got dimension ${r[a]}`)}}function Wo(n,e){const t=n,s=[];let o=0;n.length===0&&t.push(-1),o=n.length+1;for(let a=0;a<o;++a)s.push([]);const r=[];for(let a=0;a<t.length;++a){const c=t[a],i=_u(e,c);for(const l of i)r.indexOf(l)===-1&&(s[a].push(l),r.push(l))}return{path:t,steps:s}}function Uo(n){return n.every((e,t)=>e===t)}function _u(n,e){const t=[];for(let s=0;s<n.length;++s)(n[s].length===0||n[s].indexOf(e)!==-1||e===-1)&&t.push(s);return t}function Go(n,e,t=0){let s=[];if(typeof e=="number")O(n.shape[t]%e===0,()=>"Number of splits must evenly divide the axis."),s=new Array(e).fill(n.shape[t]/e);else{const o=e.reduce((a,c)=>(c===-1&&(a+=1),a),0);O(o<=1,()=>"There should be only one negative value in split array.");const r=e.indexOf(-1);if(r!==-1){const a=e.reduce((c,i)=>i>0?c+i:c);e[r]=n.shape[t]-a}O(n.shape[t]===e.reduce((a,c)=>a+c),()=>"The sum of sizes must match the size of the axis dimension."),s=e}return s}function zo(n){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${n}`}function Ho(n,e){return`indices(${n}, 0) is invalid: ${e} < 0`}function Xo(n,e,t){return`indices(${n}, 0) is invalid: ${e} >= ${t}`}function Ko(n,e){return`only one output dimension may be -1, not both ${n} and ${e}`}function jo(n,e){return`size ${n} must be non-negative, not ${e}`}function qo(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function Yo(n,e){const t=E(n),s=E(e);return`Input to reshape is a SparseTensor with ${t}
  dense values, but the requested shape requires a multiple of ${s}. inputShape=${n} outputShape= ${e}`}function Qo(n,e){const t=E(n),s=E(e);return`Input to reshape is a tensor with ${t} dense values, but the requested shape has ${s}. inputShape=${n} outputShape=${e}`}function ln(){return"segment ids must be >= 0"}function Zo(){return"segment ids are not increasing"}function Jo(n,e){return`Segment id ${n} out of range [0, ${e}), possibly because segmentIds input is not sorted.`}function er(n,e,t){return`Bad: indices[${n}] == ${e} out of range [0, ${t})`}function tr(n,e){let t=!1,s;for(n<=Yn?(s=n,t=!0):s=on(n,Math.floor(Math.sqrt(n)));!t;)s>e||s===n?t=!0:s=on(n,s+1);return s}function nr(n,e,t){const s=[],o=n.length;for(let r=0;r<o;r++)r!==e?s.push(n[r]):s.push(t);return s}function sr(n,e,t,s){const o=e.shape.length,r=n.shape.length;if(s!==0&&(s<-o||s>o))throw new Error(`Expect batchDims in the range of [-${o}, ${o}], but got ${s}`);if(s<0&&(s+=o),s>r)throw new Error(`batchDims (${s}) must be less than rank(x) (
    ${r}).`);if(t<s)throw new Error(`batchDims (${s}) must be less than or equal to axis (${t}).`);for(let d=0;d<s;++d)if(n.shape[d]!==e.shape[d])throw new Error(`x.shape[${d}]: ${n.shape[d]} should be equal to indices.shape[${d}]: ${e.shape[d]}.`);const a=n.shape[t],c=[];let i=1,l=1,u=1;for(let d=0;d<s;++d)c.push(n.shape[d]),i*=n.shape[d];for(let d=s;d<t;d++)c.push(n.shape[d]),l*=n.shape[d];for(let d=s;d<o;d++)c.push(e.shape[d]);for(let d=t+1;d<r;d++)c.push(n.shape[d]),u*=n.shape[d];return{batchSize:i,sliceSize:u,outerSize:l,dimSize:a,outputShape:c}}const Lu=Object.freeze(Object.defineProperty({__proto__:null,collectGatherOpShapeInfo:sr,computeOutShape:nr,segOpComputeOptimalWindowSize:tr},Symbol.toStringTag,{value:"Module"}));function Ce(n){try{return n.map(e=>zt(e))}catch(e){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${e}`)}}function or(n){return n.map(e=>ut(e))}const Vu=Object.freeze(Object.defineProperty({__proto__:null,ERF_A1:Do,ERF_A2:Fo,ERF_A3:Po,ERF_A4:_o,ERF_A5:Lo,ERF_P:Oo,PARALLELIZE_THRESHOLD:Yn,get RowPartitionType(){return le},SELU_SCALE:Ao,SELU_SCALEALPHA:ko,applyActivation:li,assertAndGetBroadcastShape:z,assertAxesAreInnerMostDims:de,assertParamsConsistent:wo,assignToTypedArray:Ou,axesAreInnerMostDims:ui,calculateShapes:Ht,checkEinsumDimSizes:Mo,checkPadOnDimRoundingMode:di,combineLocations:pi,combineRaggedTensorToTensorShapes:Io,complexWithEvenIndex:Nu,complexWithOddIndex:ku,computeConv2DInfo:$e,computeConv3DInfo:Xt,computeDefaultPad:hi,computeDilation2DInfo:Zs,computeOptimalWindowSize:jt,computeOutAndReduceShapes:fe,computeOutShape:Ae,computePool2DInfo:Xe,computePool3DInfo:$t,convertConv2DDataFormat:bt,decodeEinsumEquation:Vo,eitherStridesOrDilationsAreOne:Ke,expandShapeToKeepDim:me,exponent:Fu,exponents:Du,fromStringArrayToUint8:or,fromUint8ToStringArray:Ce,getAxesPermutation:se,getBroadcastDims:ze,getComplexWithIndex:Au,getEinsumComputePath:Wo,getEinsumPermutation:Bo,getFusedBiasGradient:fi,getFusedDyActivation:mi,getImageCenter:To,getInnerMostAxes:oe,getPermuted:Zn,getRaggedRank:yo,getReductionAxes:xi,getReshaped:Qn,getReshapedPermuted:Jn,getRowPartitionTypesHelper:Ro,getSliceBeginCoords:Eo,getSliceSize:No,getSparseFillEmptyRowsIndicesDenseShapeMismatch:zo,getSparseFillEmptyRowsNegativeIndexErrorMessage:Ho,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:Xo,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:qo,getSparseReshapeInputOutputMismatchErrorMessage:Qo,getSparseReshapeInputOutputMultipleErrorMessage:Yo,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:Ko,getSparseReshapeNegativeOutputDimErrorMessage:jo,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:er,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:ln,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:Zo,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:Jo,getUndoAxesPermutation:$n,isIdentityPermutation:Uo,log:gi,mergeRealAndImagArrays:pt,prepareAndValidate:Hn,prepareSplitSize:Go,segment_util:Lu,shouldFuse:Ci,slice_util:yu,splitRealAndImagArrays:Eu,stridesOrDilationsArePositive:$i,tupleValuesAreOne:bi,upcastType:ye,validateDefaultValueShape:So,validateInput:vi,validateUpdateShape:wi,warn:vt},Symbol.toStringTag,{value:"Module"}));function _e(n,e){Array.isArray(n)||(n=[n]),n.forEach(t=>{t!=null&&O(t.dtype!=="complex64",()=>`${e} does not support complex64 tensors in the CPU backend.`)})}function rr(n){const e=new Float32Array(n.length);for(let t=0;t<n.length;++t)e[t]=Math.abs(n[t]);return e}const Bu=n=>{const{x:e}=n.inputs,t=n.backend;_e(e,"abs");let s=new Float32Array(E(e.shape));const o=t.data.get(e.dataId).values;return s=rr(o),t.makeOutput(s,e.shape,e.dtype)},cv={kernelName:Js,backendName:"cpu",kernelFunc:Bu};function Y(n){return(e,t,s,o,r)=>{const a=z(e,t),c=a.length,i=M(a),l=E(a),u=ge(r,l),d=e.length,p=t.length,h=M(e),f=M(t),g=ze(e,a),x=ze(t,a);if(g.length+x.length===0)for(let m=0;m<u.length;++m)u[m]=n(s[m%s.length],o[m%o.length]);else for(let m=0;m<u.length;++m){const C=Kt(m,c,i),$=C.slice(-d);g.forEach(S=>$[S]=0);const b=dt($,d,h),v=C.slice(-p);x.forEach(S=>v[S]=0);const T=dt(v,p,f);u[m]=n(s[b],o[T])}return[u,a]}}function qt(n){const{inputs:e,backend:t}=n,{real:s,imag:o}=e,r=t.data.get(s.dataId).values,a=t.data.get(o.dataId).values,c=t.makeTensorInfo(s.shape,"complex64"),i=t.data.get(c.dataId);return i.complexTensorInfos={real:t.makeTensorInfo(s.shape,"float32",r),imag:t.makeTensorInfo(o.shape,"float32",a)},c}const lv={kernelName:eo,backendName:"cpu",kernelFunc:qt};function un(n,e,t="float32"){if(t==="complex64"){const o=un(n,e,"float32"),r=un(n,e,"float32");return qt({inputs:{real:o,imag:r},backend:n})}const s=He(E(e),t);return n.makeTensorInfo(e,t,s)}function dn(n){const{inputs:e,backend:t}=n,{x:s}=e;return t.incRef(s.dataId),{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}const uv={kernelName:to,backendName:"cpu",kernelFunc:dn};function ar(n){const{inputs:e,backend:t}=n,{input:s}=e,o=t.data.get(s.dataId).complexTensorInfos.real,r=t.data.get(o.dataId).values;return t.makeTensorInfo(o.shape,o.dtype,r)}const dv={kernelName:no,backendName:"cpu",kernelFunc:ar};function ir(n,e,t,s){if(s==="int32"){const o=Int32Array.from(n);return[e,"int32",o]}if(s==="bool"){const o=Ii([0],t),[r,a]=Y((c,i)=>c!==i?1:0)(e,[],n,o,"bool");return[a,"bool",r]}throw new Error(`Error in Cast: failed to cast ${t} to ${s}`)}function ht(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{dtype:r}=s;if(r==="complex64"){if(o.dtype==="complex64")return dn({inputs:{x:o},backend:t});const u=un(t,o.shape,o.dtype),d=ht({inputs:{x:o},backend:t,attrs:{dtype:"float32"}}),p=qt({inputs:{real:d,imag:u},backend:t});return t.disposeIntermediateTensorInfo(u),t.disposeIntermediateTensorInfo(d),p}if(o.dtype==="complex64"){const u=ar({inputs:{input:o},backend:t}),d=ht({inputs:{x:u},backend:t,attrs:{dtype:r}});return t.disposeIntermediateTensorInfo(u),d}if(!oo(o.dtype,r)){const u=dn({inputs:{x:o},backend:t});return{dataId:u.dataId,shape:u.shape,dtype:r}}const a=t.data.get(o.dataId).values,[c,i,l]=ir(a,o.shape,o.dtype,r);return t.makeTensorInfo(c,i,l)}const pv={kernelName:so,backendName:"cpu",kernelFunc:ht};function J(n,e,t,s){return t==null?({inputs:o,backend:r})=>{const{a,b:c}=o,i=r;_e([a,c],n);const l=i.data.get(a.dataId).values,u=i.data.get(c.dataId).values,d=a.dtype==="string"?Ce(l):l,p=a.dtype==="string"?Ce(u):u,h=s||a.dtype,[f,g]=e(a.shape,c.shape,d,p,h);return i.makeTensorInfo(g,h,f)}:({inputs:o,backend:r})=>{const{a,b:c}=o,i=r;if(a.dtype==="complex64"||c.dtype==="complex64"){const l=ht({inputs:{x:a},backend:i,attrs:{dtype:"complex64"}}),u=i.data.get(l.dataId),d=u.complexTensorInfos.real,p=u.complexTensorInfos.imag,h=i.data.get(d.dataId).values,f=i.data.get(p.dataId).values,g=ht({inputs:{x:c},backend:i,attrs:{dtype:"complex64"}}),x=i.data.get(g.dataId),m=x.complexTensorInfos.real,C=x.complexTensorInfos.imag,$=i.data.get(m.dataId).values,b=i.data.get(C.dataId).values,[v,T,S]=t(a.shape,c.shape,h,f,$,b),I=i.makeTensorInfo(S,"float32",v),A=i.makeTensorInfo(S,"float32",T),k=qt({inputs:{real:I,imag:A},backend:i});return i.disposeIntermediateTensorInfo(l),i.disposeIntermediateTensorInfo(g),i.disposeIntermediateTensorInfo(I),i.disposeIntermediateTensorInfo(A),k}else{const l=i.data.get(a.dataId).values,u=i.data.get(c.dataId).values,d=s||a.dtype,[p,h]=e(a.shape,c.shape,l,u,d);return i.makeTensorInfo(h,d,p)}}}function es(n){return(e,t,s,o,r,a)=>{const c=z(e,t),i=E(c),l=c.length,u=M(c),d=ge("float32",i),p=ge("float32",i),h=ze(e,c),f=ze(t,c),g=pt(s,o),x=pt(r,a),m=e.length,C=M(e),$=t.length,b=M(t);if(h.length+f.length===0)for(let v=0;v<d.length;v++){const T=v%g.length,S=v%x.length,I=n(g[T*2],g[T*2+1],x[S*2],x[S*2+1]);d[v]=I.real,p[v]=I.imag}else for(let v=0;v<d.length;v++){const T=Kt(v,l,u),S=T.slice(-m);h.forEach(P=>S[P]=0);const I=dt(S,m,C),A=T.slice(-$);f.forEach(P=>A[P]=0);const k=dt(A,$,b),F=n(g[I*2],g[I*2+1],x[k*2],x[k*2+1]);d[v]=F.real,p[v]=F.imag}return[d,p,c]}}const cr=Y(((n,e)=>n+e)),Mu=es(((n,e,t,s)=>({real:n+t,imag:e+s}))),Wu=J(bn,cr,Mu),hv={kernelName:bn,backendName:"cpu",kernelFunc:Wu};function Uu(n,e,t,s,o){const r=E(s),a=He(o,t);for(let c=0;c<n.length;c++){const i=n[c];if(i<0)throw new Error("Input x must be non-negative!");i>=o||(r>0?a[i]+=e[c]:a[i]+=1)}return a}function Gu(n,e,t,s=!1){const o=n.shape[0],r=n.shape[1],a=j([o,t],e.dtype);for(let c=0;c<o;c++)for(let i=0;i<r;i++){const l=n.get(c,i);if(l<0)throw new Error("Input x must be non-negative!");l>=t||(s?a.set(1,c,l):e.size>0?a.set(a.get(c,l)+e.get(c,i),c,l):a.set(a.get(c,l)+1,c,l))}return a}const lr=Y(((n,e)=>n&e)),zu=J(vn,lr),fv={kernelName:vn,backendName:"cpu",kernelFunc:zu};function xe(n){return(e,t,s)=>{const o=B(t,e.length);for(let r=0;r<e.length;++r)o[r]=n(e[r],s);return o}}function ur(n,e,t){const s=xe(e);return Se(n,s,t)}function Se(n,e,t){return({inputs:s,attrs:o,backend:r})=>{const{x:a}=s;_e(a,n);const c=r,i=c.data.get(a.dataId).values;let l;if(a.dtype==="string"){if(!Array.isArray(i))throw new Error("String tensor's value was not an instance of Array");l=Ce(i)}else l=i;const u=t||a.dtype,d=e(l,u,o);return c.makeTensorInfo(a.shape,u,d)}}const dr=xe(n=>Math.ceil(n)),Hu=Se(wn,dr),mv={kernelName:wn,backendName:"cpu",kernelFunc:Hu};function Xu(n,e,t,s){const o=B(t,E(e));if(s&&t!=="string"){let r=0;n.forEach(a=>{const c=E(a.shape);o.set(a.vals,r),r+=c})}else{let r=0;n.forEach(a=>{const c=t==="string"?Ce(a.vals):a.vals;let i=0;for(let l=0;l<a.shape[0];++l){const u=l*e[1]+r;for(let d=0;d<a.shape[1];++d)o[u+d]=c[i++]}r+=a.shape[1]})}return o}const pr=Y((n,e)=>n===e?1:0),Ku=J(In,pr,null,"bool"),xv={kernelName:In,backendName:"cpu",kernelFunc:Ku};const hr=xe(n=>Math.exp(n)),ju=Se(Rn,hr,"float32"),gv={kernelName:Rn,backendName:"cpu",kernelFunc:ju};const fr=xe(n=>Math.expm1(n)),qu=Se(yn,fr),Cv={kernelName:yn,backendName:"cpu",kernelFunc:qu};const mr=xe(n=>Math.floor(n)),Yu=Se(Sn,mr),$v={kernelName:Sn,backendName:"cpu",kernelFunc:Yu};const xr=Y((n,e)=>Math.floor(n/e)),Qu=J(Tn,xr,null,"int32"),bv={kernelName:Tn,backendName:"cpu",kernelFunc:Qu};function Zu(n,e,t,s,o,r,a,c,i){const l=j([s,r],t);for(let u=0;u<s;u++){const d=[];let p=0;for(let h=0;h<o;h++){const f=n[u*o+h];p+=f*a[h],d.push(f)}if(p<0||p>=i/r)throw new Error(`Invalid indices: ${d} does not index into ${c}`);for(let h=0;h<r;h++)l.values[u*r+h]=e.get(...e.indexToLoc(p*r+h))}return l}function Ju(n,e,t){const s=j(t,n.dtype);for(let o=0;o<s.size;++o){const a=s.indexToLoc(o).slice(),c=a[0],i=a[2],l=e.locToIndex([c,i]);a[2]=e.values[l];const u=n.locToIndex(a);0<=u&&u<n.values.length&&(s.values[o]=n.values[u])}return s}const gr=Y((n,e)=>n>e?1:0),ed=J(En,gr,null,"bool"),vv={kernelName:En,backendName:"cpu",kernelFunc:ed};const Cr=Y((n,e)=>n>=e?1:0),td=J(Nn,Cr,null,"bool"),wv={kernelName:Nn,backendName:"cpu",kernelFunc:td};const $r=Y((n,e)=>n<e?1:0),nd=J(kn,$r,null,"bool"),Iv={kernelName:kn,backendName:"cpu",kernelFunc:nd};const br=Y((n,e)=>n<=e?1:0),sd=J(An,br,null,"bool"),Rv={kernelName:An,backendName:"cpu",kernelFunc:sd};function od(n,e,t){const s=(e-n)/(t-1),o=He(t,"float32");o[0]=n;for(let r=1;r<o.length;r++)o[r]=o[r-1]+s;return o}const vr=xe(n=>Math.log(n)),rd=Se(On,vr),yv={kernelName:On,backendName:"cpu",kernelFunc:rd};function ad(n,e,t,s){const o=ge(s,E(t));for(let r=0;r<o.length;++r){const a=r*e;let c=n[a];for(let i=0;i<e;++i){const l=n[a+i];(Number.isNaN(l)||l>c)&&(c=l)}o[r]=c}return o}const wr=Y(((n,e)=>Math.max(n,e))),id=J(Dn,wr),Sv={kernelName:Dn,backendName:"cpu",kernelFunc:id};const Ir=Y(((n,e)=>Math.min(n,e))),cd=J(Fn,Ir),Tv={kernelName:Fn,backendName:"cpu",kernelFunc:cd};const ts=Y(((n,e)=>n*e)),ld=es(((n,e,t,s)=>({real:n*t-e*s,imag:n*s+e*t}))),ud=J(Pn,ts,ld),Ev={kernelName:Pn,backendName:"cpu",kernelFunc:ud};function Rr(n,e,t){const s=je(-1,t);return ts([],e,s,n,t)}function dd(n){const{inputs:e,backend:t}=n,{x:s}=e;_e(s,"neg");const o=t.data.get(s.dataId).values,[r,a]=Rr(o,s.shape,s.dtype);return t.makeTensorInfo(a,s.dtype,r)}const Nv={kernelName:ro,backendName:"cpu",kernelFunc:dd};const yr=Y(((n,e)=>n!==e?1:0)),pd=J(_n,yr,null,"bool"),kv={kernelName:_n,backendName:"cpu",kernelFunc:pd};function Sr(n,e,t,s,o){const r=e.length,a=E(e),c=M(e),i=M(o),l=ge(t,E(o));for(let u=0;u<a;++u){const d=Kt(u,r,c),p=new Array(d.length);for(let f=0;f<p.length;f++)p[f]=d[s[f]];const h=dt(p,r,i);l[h]=n[u]}return l}function Tr(n){const{inputs:e,attrs:t,backend:s}=n,{x:o}=e,{perm:r}=t;_e(o,"transpose");const a=o.shape.length,c=new Array(a);for(let d=0;d<c.length;d++)c[d]=o.shape[r[d]];const i=s.data.get(o.dataId).values,l=Sr(i,o.shape,o.dtype,r,c);return{dataId:s.write(l,c,o.dtype),shape:c,dtype:o.dtype}}const Av={kernelName:ao,backendName:"cpu",kernelFunc:Tr};function Er(n,e,t,s){const[o,r]=fe(n,s),a=ye(e,"int32"),c=He(E(o),a),i=E(r);for(let l=0;l<c.length;++l){const u=l*i;let d=1;for(let p=0;p<i;++p)d*=t[u+p];c[l]=d}return{outVals:c,outShape:o,outDtype:a}}function hd(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,keepDims:a}=s;_e(o,"prod");const c=o.shape.length,i=X(r,o.shape),l=se(i,c);let u=i,d=o;const p=[];l!=null&&(d=Tr({inputs:{x:o},backend:t,attrs:{perm:l}}),p.push(d),u=oe(u.length,c));const h=t.data.get(d.dataId).values,{outVals:f,outShape:g,outDtype:x}=Er(d.shape,d.dtype,h,u);let m=g;return a&&(m=me(g,i)),p.forEach(C=>t.disposeIntermediateTensorInfo(C)),t.makeTensorInfo(m,x,f)}const Ov={kernelName:io,backendName:"cpu",kernelFunc:hd};function fd(n,e,t){n.forEach((s,o)=>{if(s<0||s>=t){const r=Kt(o,e.length,M(e)).join(",");throw new Error(`indices[${r}] = ${s} is not in [0, ${t})`)}})}function md(n,e){for(let t=0;t<n.length;++t){const s=n[t],o=t===n.length-1?e:n[t+1].length;if(s.length===0)throw new Error("Ragged splits may not be empty");if(s[0]<0)throw new Error("Ragged splits must be non-negative");if(s[s.length-1]>o)throw new Error("Ragged splits must not point past values");for(let r=1;r<s.length;++r)if(s[r-1]>s[r])throw new Error("Ragged splits must be sorted in ascending order")}}function xd(n,e,t,s){const o=[];let r=0;const a=e.length-1+t.length,c=new Array(a).fill(null).map(()=>[0]);md(t,s);let i=1;for(let l=0;l<e.length-1;++l){i*=e[l];const u=e[l+1];for(let d=1;d<i+1;++d)c[l].push(d*u)}for(let l=0;l<n.length;++l){let u=n[l],d=n[l]+1;for(let p=0;p<t.length;++p){const h=t[p],f=p+e.length-1;if(f>=0){const g=c[f],x=g[g.length-1]-h[u];for(let m=u;m<d;++m)c[f].push(h[m+1]+x)}u=h[u],d=h[d]}d!==u&&(o.push([u,d]),r+=d-u)}return{outSplits:c,valueSlices:o,numValues:r}}function gd(n){const e=[];for(let t=0;t<n.length;++t){const s=n[t].length,o=B("int32",s);e.push(o),n[t].forEach((r,a)=>o[a]=r)}return e}function Rs(n,e){const t=n.slice(0,e);for(;t.length<e;)t.push(1);for(let s=e;s<n.length;s++)t[e-1]*=n[s];return t}function Cd(n,e,t,s,o,r){const a=Rs(e,2)[1],c=Rs(r,2)[1];let i=0;for(const l of t)for(let u=l[0];u<l[1];++u){for(let d=0;d<s;++d)o[i*c+d]=n[u*a+d];++i}}function $d(n,e,t,s,o){const r=e.slice();r[0]=o;const a=B(t,E(r)),c=n.length,i=c===0?0:c/e[0];return Cd(n,e,s,i,a,r),[a,r]}function bd(n,e,t,s,o,r,a,c){if(n.length===0)throw new Error("paramsNestedSplits must be non empty");if(e[0].length===0)throw new Error("Split tensors must not be scalars");const i=e[0][0]-1;if(fd(r,a,i),s.length===0)throw new Error("params.rank must be nonzero");const l=s[0],{outSplits:u,valueSlices:d,numValues:p}=xd(r,a,n,l),h=gd(u),f=$d(t,s,o,d,p);return[h,f[0],f[1]]}const ys=2147483647;function vd(n,e,t,s,o,r,a){if(e.length>1)throw new Error("starts must be a scalar or vector");if(o.length>1)throw new Error("limits must be a scalar or vector");if(a.length>1)throw new Error("deltas must be a scalar or vector");const c=e.length===0,i=o.length===0,l=a.length===0,u=[];c||u.push(e[0]),i||u.push(o[0]),l||u.push(a[0]);for(let x=1;x<u.length;++x)if(u[x]!==u[x-1])throw new Error("starts, limits, and deltas must have the same shape");const d=u.length===0?1:u[0],p=B("int32",d+1);p[0]=0;for(let x=0;x<d;++x){const m=c?n[0]:n[x],C=i?s[0]:s[x],$=l?r[0]:r[x];if($===0)throw new Error("Requires delta != 0");let b;if($>0&&C<m||$<0&&C>m)b=0;else if(b=Math.ceil(Math.abs((C-m)/$)),b>ys)throw new Error(`Requires ((limit - start) / delta) <= ${ys}`);p[x+1]=p[x]+b}const h=p[d],f=B(t,h);let g=0;for(let x=0;x<d;++x){const m=p[x+1]-p[x];let C=c?n[0]:n[x];const $=l?r[0]:r[x];for(let b=0;b<m;++b)f[g++]=C,C+=$}return[p,f]}var ae=le;class Mt{constructor(e,t,s,o,r,a,c,i,l,u){this.shape=e,this.shapeShape=t,this.values=s,this.valuesShape=o,this.valuesDType=r,this.defaultValue=a,this.defaultValueShape=c,this.rowPartitionValues=i,this.rowPartitionValuesShapes=l,this.rowPartitionTypes=Ro(u),this.raggedRank=yo(this.rowPartitionTypes)}getRowPartitionTypeByDimension(e){return this.rowPartitionTypes[0]===ae.FIRST_DIM_SIZE?this.rowPartitionTypes[e+1]:this.rowPartitionTypes[e]}getRowPartitionTensor(e){return this.rowPartitionTypes[0]===ae.FIRST_DIM_SIZE?this.rowPartitionValues[e+1]:this.rowPartitionValues[e]}getMaxWidth(e){const t=this.getRowPartitionTensor(e-1);switch(this.getRowPartitionTypeByDimension(e-1)){case ae.VALUE_ROWIDS:return Mt.getMaxWidthValueRowID(t);case ae.ROW_SPLITS:return Mt.getMaxWidthRowSplit(t);default:throw new Error(`Cannot handle partition type ${ae[this.getRowPartitionTypeByDimension(e-1)]}`)}}static getMaxWidthRowSplit(e){const t=e.length;if(t===0||t===1)return 0;let s=0;for(let o=0;o<t-1;++o){const r=e[o+1]-e[o];r>s&&(s=r)}return s}static getMaxWidthValueRowID(e){const t=e.length;if(t===0)return 0;let s=0,o=e[0],r=0;for(let a=1;a<t;++a){const c=e[a];c!==o&&(o=c,r=Math.max(a-s,r),s=a)}return Math.max(t-s,r)}tensorShapeFromTensor(e,t,s=!0){if(t.length===0){if(e[0]===-1)return[];throw new Error("The only valid scalar shape tensor is the fully unknown shape specified as -1.")}return Ts(e,s)}calculateOutputSize(e){const t=this.valuesShape,s=this.defaultValueShape;So(s,t);const o=this.tensorShapeFromTensor(this.shape,this.shapeShape),a=Io(this.raggedRank,o,t);a[0]<0&&(a[0]=e);for(let c=1;c<=this.raggedRank;++c)a[c]<0&&(a[c]=this.getMaxWidth(c));return a}calculateFirstParentOutputIndex(e,t,s){const o=Math.min(e,s),r=[];let a=0;for(let c=0;c<o;++c,a+=t)r.push(a);for(let c=o;c<e;++c)r.push(-1);return O(r.length===e,()=>"Final length of result must be equal to firstDimension."),r}calculateOutputIndexRowSplit(e,t,s,o){const r=e.length,a=[];for(let c=0;c<r-1;++c){const i=e[c+1]-e[c];let l=Math.min(o,i),u=t[c];u===-1&&(l=0);for(let d=0;d<l;++d)a.push(u),u+=s;for(let d=0;d<i-l;++d)a.push(-1)}if(r>0&&a.length!==e[r-1])throw new Error("Invalid row split size.");return a}calculateOutputIndexValueRowID(e,t,s,o){const r=e.length,a=[];if(r===0)return[];let c=0,i=e[0];if(i>=t.length)throw new Error(`Got currentValueRowId=${i}, which is not less than ${t.length}`);let l=t[i];a.push(l);for(let u=1;u<r;++u){const d=e[u];if(d===i)l>=0&&(++c,c<o?l+=s:l=-1);else{if(c=0,i=d,d>=t.length)throw new Error(`Got nextValueRowId=${d} which is not less than ${t.length}`);l=t[d]}a.push(l)}if(a.length!==e.length)throw new Error("Invalid row ids.");return a}calculateOutputIndex(e,t,s,o){const r=this.getRowPartitionTensor(e),a=this.getRowPartitionTypeByDimension(e);switch(a){case ae.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(r,t,s,o);case ae.ROW_SPLITS:if(r.length-1>t.length)throw new Error(`Row partition size is greater than output size: ${r.length-1} > ${t.length}`);return this.calculateOutputIndexRowSplit(r,t,s,o);default:throw new Error(`Unsupported partition type: ${ae[a]}`)}}getFirstDimensionSize(){const e=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw new Error("No row_partition_types given.");const t=this.rowPartitionTypes[0];switch(t){case ae.FIRST_DIM_SIZE:return e[0];case ae.VALUE_ROWIDS:throw new Error("Cannot handle VALUE_ROWIDS in first dimension.");case ae.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw new Error(`Cannot handle type ${ae[t]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw new Error("Invalid first partition input. Tensor requires at least one element.");const t=this.getFirstDimensionSize(),s=this.calculateOutputSize(t),o=new Array(this.raggedRank+1);o[o.length-1]=1;for(let i=o.length-2;i>=0;--i)o[i]=o[i+1]*s[i+1];const r=Ts(s,!1),a=B(this.valuesDType,E(r));if(o[0]*s[0]>0){let i=this.calculateFirstParentOutputIndex(t,o[0],s[0]);for(let l=1;l<=this.raggedRank;++l)i=this.calculateOutputIndex(l-1,i,o[l],s[l]);this.setOutput(this.raggedRank,i,a,r)}return[r,a]}setOutput(e,t,s,o){if(s.length===0)return;const r=this.values,a=s;let c=o.slice();c=c.slice(e+1);const i=E(c),l=t.length;let u=this.defaultValue;if(u.length!==i&&u.length!==1){const f=this.defaultValueShape;co(()=>{const g=Ri(u,f);u=yi(g,c).dataSync()})}let d=0,p=0,h=0;for(let f=0;f<=l;++f){let g=f<l?t[f]:-1;if(g===h){++h;continue}if(p<h){const x=r.subarray(d*i),m=a.subarray(p*i),C=(h-p)*i;Ss(m,x,C)}if(f>=l){const x=s.length;g=Math.floor(x/i)}if(g>h)if(this.defaultValue.length===1)a.subarray(h*i,g*i).fill(this.defaultValue[0]),h=g;else for(;g>h;){const x=a.slice(h*i);Ss(x,u,i),++h}g<0?(d=f+1,p=h):(d=f,p=h,h=p+1)}}}function Ss(n,e,t){for(let s=0;s<t;s++)n[s]=e[s]}function Ts(n,e){const t=[];for(let s of n){if(s<0){if(!e)throw new Error(`Dimension ${s} must be >= 0`);if(s<-1)throw new Error(`Dimension ${s} must be >= -1`);s=-1}t.push(s)}return t}function wd(n,e,t,s,o,r,a,c,i,l){return new Mt(n,e,t,s,o,r,a,c,i,l).compute()}function Id(n,e,t,s){const o=n===e,r=n<e&&t<0,a=e<n&&t>1;if(o||r||a)return He(0,s);const c=Math.abs(Math.ceil((e-n)/t)),i=He(c,s);e<n&&t===1&&(t=-1),i[0]=n;for(let l=1;l<i.length;l++)i[l]=i[l-1]+t;return i}const Nr=xe(n=>1/Math.sqrt(n)),Rd=Se(Ln,Nr),Dv={kernelName:Ln,backendName:"cpu",kernelFunc:Rd};function yd(n,e,t,s,o,r,a,c,i,l){const u=[s/o,o],d=n.values,p=e.values;if(s===0)return j(t,e.dtype);const h=i instanceof rn?i:j(u,e.dtype);typeof i=="string"||typeof i=="number"?h.values.fill(i):typeof i=="boolean"&&h.values.fill(+i);for(let f=0;f<r;f++){const g=[];let x=0;for(let m=0;m<a;m++){const C=d[f*a+m];g.push(C),x+=C*c[m]}if(x<0||x>=s/o)throw new Error(`Invalid indices: ${g} does not index into ${t}`);for(let m=0;m<o;m++)l?h.values[x*o+m]+=p[f*o+m]:h.values[x*o+m]=e.rank===0?p[0]:p[f*o+m]}return h}const Sd=xe(n=>1/(1+Math.exp(-n))),Td=ur(Vn,n=>1/(1+Math.exp(-n))),Fv={kernelName:Vn,backendName:"cpu",kernelFunc:Td};function kr(n,e,t,s,o){const r=Kn(s,e,t),a=E(t),c=M(s);if(r){const d=jn(e,c);return o==="string"?n.slice(d,d+a):n.subarray(d,d+a)}const i=o==="string"?Ce(n):n,l=j(s,o,i),u=j(t,o);for(let d=0;d<u.size;++d){const p=u.indexToLoc(d),h=p.map((f,g)=>f+e[g]);u.set(l.get(...h),...p)}return o==="string"?or(u.values):u.values}function Ed(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{begin:r,size:a}=s;_e(o,"slice");const[c,i]=qn(o,r,a);Xn(o,c,i);const l=t.data.get(o.dataId).values,u=kr(l,c,i,o.shape,o.dtype);return t.makeTensorInfo(i,o.dtype,u)}const Pv={kernelName:lo,backendName:"cpu",kernelFunc:Ed};function Nd(n,e,t,s,o,r,a){const c=e[0],i=r[0],l=new Array(i),u=new Array(c),d=e[1];if(i===0){if(c!==0)throw new Error(zo(c));const x=B(t,0),m=B(o,0);return[x,[0,d],m,l,u]}let p=!0,h=0;const f=new Array(i).fill(0);for(let x=0;x<c;++x){const m=n[x*d];if(m<0)throw new Error(Ho(x,m));if(m>=i)throw new Error(Xo(x,m,i));++f[m],p=p&&m>=h,h=m}let g=!0;for(let x=0;x<i;++x){const m=f[x]===0;l[x]=m,g=g&&!m,f[x]=Math.max(f[x],1),x>0&&(f[x]+=f[x-1])}if(g&&p){const x=n,m=s;for(let C=0;C<c;++C)u[C]=C;return[x,[c,d],m,l,u]}else{const x=f[i-1],m=B(t,x*d),C=B(o,x),$=new Array(i).fill(0);for(let b=0;b<c;++b){const v=n[b*d],T=$[v],S=(v===0?0:f[v-1])+T;$[v]++;for(let I=0;I<d;++I)m[S*d+I]=n[b*d+I];C[S]=s[b],u[b]=S}for(let b=0;b<i;++b)if($[b]===0){const T=b===0?0:f[b-1];m[T*d+0]=b;for(let S=1;S<d;++S)m[T*d+S]=0;C[T]=a}return[m,[x,d],C,l,u]}}function kd(n,e,t,s,o){const r=E(s),a=e[0],c=o.length,i=[];let l=1,u=-1;for(let x=0;x<c;++x){const m=o[x];if(m===-1){if(u!==-1)throw new Error(Ko(u,x));u=x,i.push(1)}else{if(m<0)throw new Error(jo(x,m));l*=m,i.push(m)}}if(u!==-1){if(l<=0)throw new Error(qo());const x=Math.trunc(r/l);if(l*x!==r)throw new Error(Yo(s,i));i[u]=x}if(E(i)!==r)throw new Error(Qo(s,i));const p=s.length,h=[];if(p>0){h[p-1]=1;for(let x=p-2;x>=0;--x)h[x]=h[x+1]*s[x+1]}const f=[];if(c>0){f[c-1]=1;for(let x=c-2;x>=0;--x)f[x]=f[x+1]*i[x+1]}const g=B(t,a*c);for(let x=0;x<a;++x){let m=0;for(let C=0;C<p;++C)m+=n[x*p+C]*h[C];for(let C=0;C<c;++C)g[x*c+C]=Math.trunc(m/f[C]),m%=f[C]}return[g,[a,c],i]}function Ad(n,e,t,s,o,r=!1,a=0){const c=s.length,i=[e[0],n.length/e[0]],l=i[1],d=c>0?o[c-1]+1:0;if(d<0)throw new Error(ln());const p=e.slice();p[0]=d;const h=p.reduce(($,b)=>$*b,1),f=B(t,h);if(c===0)return d>0&&f.fill(a),[f,p];if(d<=0)throw new Error(ln());let g=0,x=1,m=0,C=o[g];for(;;){let $=0;if(x<c){if($=o[x],C===$){++x;continue}if(C>=$)throw new Error(Zo())}if(C<0||C>=d)throw new Error(Jo(C,d));C>m&&f.fill(a,m*l,C*l);for(let b=g;b<x;++b){const v=s[b];if(v<0||v>=i[0])throw new Error(er(b,s[b],i[0]));for(let T=0;T<l;T++)f[C*l+T]+=n[v*l+T]}if(r)for(let b=0;b<l;b++)f[C*l+b]/=x-g;if(g=x,++x,m=C+1,C=$,x>c)break}return m<d&&f.fill(a,m*l,d*l),[f,p]}const Od=xe(n=>Math.sqrt(n)),Dd=ur(Bn,n=>Math.sqrt(n)),_v={kernelName:Bn,backendName:"cpu",kernelFunc:Dd};const Ar=Y(((n,e)=>{const t=n-e;return t*t})),Fd=J(Mn,Ar),Lv={kernelName:Mn,backendName:"cpu",kernelFunc:Fd};const Or=xe((n,e)=>{const{pattern:t,replaceGlobal:s,rewrite:o}=e;return n.replace(new RegExp(t,s?"g":""),o)}),Pd=Se(Wn,Or),Vv={kernelName:Wn,backendName:"cpu",kernelFunc:Pd};function _d(n,e,t,s){const o=j(n,e.dtype);for(let r=0;r<o.size;r++){const a=o.indexToLoc(r),c=new Array(a.length);for(let i=0;i<c.length;i++)c[i]=a[i]*t[i]+s[i];o.set(e.get(...c),...a)}return o}class Ld{constructor(e,t,s,o,r,a){this.separator=ut(e),this.nGramWidths=t,this.leftPad=ut(s),this.rightPad=ut(o),this.padWidth=r,this.preserveShort=a}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){const s=this.getPadWidth(t);return Math.max(0,e+2*s-t+1)}createNGrams(e,t,s,o,r,a){for(let c=0;c<r;++c){const i=this.getPadWidth(a),l=Math.max(0,i-c),u=Math.max(0,i-(r-(c+1))),d=a-(l+u),p=t+(l>0?0:c-i);let h=0;h+=l*this.leftPad.length;for(let C=0;C<d;++C)h+=e[p+C].length;h+=u*this.rightPad.length;const f=l+u+d-1;h+=f*this.separator.length,s[o+c]=new Uint8Array(h);const g=s[o+c];let x=0;const m=C=>C.forEach($=>g[x++]=$);for(let C=0;C<l;++C)m(this.leftPad),m(this.separator);for(let C=0;C<d-1;++C)m(e[p+C]),m(this.separator);if(d>0){m(e[p+d-1]);for(let C=0;C<u;++C)m(this.separator),m(this.rightPad)}else{for(let C=0;C<u-1;++C)m(this.rightPad),m(this.separator);m(this.rightPad)}}}compute(e,t){const s=e.length,o=t.length;if(o>0){let i=t[0];if(i!==0)throw new Error(`First split value must be 0, got ${i}`);for(let l=1;l<o;++l){let u=t[l]>=i;if(u=u&&t[l]<=s,!u)throw new Error(`Invalid split value ${t[l]}, must be in [${i}, ${s}]`);i=t[l]}if(i!==s)throw new Error(`Last split value must be data size. Expected ${s}, got ${i}`)}const r=o-1,a=B("int32",o);if(s===0||o===0){const i=new Array(s);for(let l=0;l<=r;++l)a[l]=0;return[i,a]}a[0]=0;for(let i=1;i<=r;++i){const l=t[i]-t[i-1];let u=0;this.nGramWidths.forEach(d=>{u+=this.getNumNGrams(l,d)}),this.preserveShort&&l>0&&u===0&&(u=1),a[i]=a[i-1]+u}const c=new Array(a[r]);for(let i=0;i<r;++i){const l=t[i];let u=a[i];if(this.nGramWidths.forEach(d=>{const p=t[i+1]-t[i],h=this.getNumNGrams(p,d);this.createNGrams(e,l,c,u,h,d),u+=h}),this.preserveShort&&u===a[i]){const d=t[i+1]-t[i];if(d===0)continue;const p=d+2*this.padWidth;this.createNGrams(e,l,c,u,1,p)}}return[c,a]}}function Vd(n,e,t,s,o,r,a,c){return new Ld(t,s,o,r,a,c).compute(n,e)}function Bd(n,e,t,s){if(!n.length)return;if(e.length===0){for(let r=0;r<n.length;++r)s.push(n.subarray(r,r+1));return}if(e.length===1){const r=e[0];let a=n.indexOf(r);for(;a!==-1;){const c=n.subarray(0,a);(!t||c.length!==0)&&s.push(c),n=n.subarray(a+1),a=n.indexOf(r)}(!t||n.length!==0)&&s.push(n);return}let o=0;for(let r=0;r<n.length+1;r++)if(r===n.length||e.indexOf(n[r])!==-1){const a=n.subarray(o,r);(!t||a.length!==0)&&s.push(a),o=r+1}}function Md(n,e,t){const s=n.length,o=[];let r=0,a=0;const c=new Array(s);for(let p=0;p<s;++p){const h=o.length;Bd(n[p],e,t,o);const f=o.length-h;c[p]=f,r+=f,a=Math.max(a,f)}const i=B("int32",r*2),l=new Array(r),u=[s,a];let d=0;for(let p=0;p<s;++p)for(let h=0;h<c[p];++h)i[d*2]=p,i[d*2+1]=h,l[d]=o[d],++d;return[i,l,u]}function Wd(n,e){const t=B("int32",n.length);for(let s=0;s<n.length;++s)t[s]=Si(n[s]).modulo(e).getLowBitsUnsigned();return t}const Dr=Y(((n,e)=>n-e)),Ud=es(((n,e,t,s)=>({real:n-t,imag:e-s}))),Gd=J(Un,Dr,Ud),Bv={kernelName:Un,backendName:"cpu",kernelFunc:Gd};function zd(n,e){const t=new Array(n.rank);for(let o=0;o<t.length;o++)t[o]=n.shape[o]*e[o];const s=j(t,n.dtype);for(let o=0;o<s.values.length;++o){const r=s.indexToLoc(o),a=new Array(n.rank);for(let i=0;i<a.length;i++)a[i]=r[i]%n.shape[i];const c=n.locToIndex(a);s.values[o]=n.values[c]}return s}const rt=(n,e)=>{const t=e.value-n.value;return t===0?n.index-e.index:t};function Fr(n,e,t=0,s=n.length-1){for(;s>t;){if(s-t>600){const c=s-t+1,i=e-t+1,l=Math.log(c),u=.5*Math.exp(2*l/3),d=.5*Math.sqrt(l*u*(c-u)/c)*Math.sign(i-c/2),p=Math.max(t,Math.floor(e-i*u/c+d)),h=Math.min(s,Math.floor(e+(c-i)*u/c+d));Fr(n,e,p,h)}const o=n[e];let r=t,a=s;for(ot(n,t,e),rt(n[s],o)>0&&ot(n,t,s);r<a;){for(ot(n,r,a),r++,a--;rt(n[r],o)<0;)r=r+1;for(;rt(n[a],o)>0;)a=a-1}rt(n[t],o)===0?ot(n,t,a):(a=a+1,ot(n,a,s)),a<=e&&(t=a+1),e<=a&&(s=a-1)}}function Hd(n,e,t,s,o){const r=e[e.length-1],[a,c]=[n.length/r,r],i=ge(t,a*s),l=ge("int32",a*s);for(let d=0;d<a;d++){const p=d*c,h=n.subarray(p,p+c);let f=new Array(h.length);h.forEach((C,$)=>f[$]={value:C,index:$}),s<f.length&&(Fr(f,s),f=f.slice(0,s)),o&&f.sort(rt);const g=d*s,x=i.subarray(g,g+s),m=l.subarray(g,g+s);for(let C=0;C<s;C++)x[C]=f[C].value,m[C]=f[C].index}const u=e.slice();return u[u.length-1]=s,[j(u,t,i),j(u,"int32",l)]}function Xd(n,e,t,s){const o=X(e,t)[0],r=[1,t[0],1];for(let f=0;f<o;f++)r[0]*=t[f];r[1]=t[o];for(let f=o+1;f<t.length;f++)r[2]*=t[f];const a=new Map,c=new Int32Array(t[o]),i=new rn(r,s,n),l=[],u=r[0]===1&&r[2]===1;for(let f=0;f<t[o];f++){let g;if(u)g=n[f].toString();else{const m=[];for(let C=0;C<r[0];C++)for(let $=0;$<r[2];$++)m.push(i.get(C,f,$));g=m.join(",")}const x=a.get(g);if(x!=null)c[f]=x;else{const m=a.size;a.set(g,m),c[f]=m,l.push(f)}}const d=r.slice();d[1]=a.size;const p=new rn(d,s);l.forEach((f,g)=>{for(let x=0;x<r[0];x++)for(let m=0;m<r[2];m++)p.set(i.get(x,f,m),x,g,m)});const h=t.slice();return h[o]=d[1],{outputValues:p.values,outputShape:h,indices:c}}const Kd=Object.freeze(Object.defineProperty({__proto__:null,addImpl:cr,bincountImpl:Uu,bincountReduceImpl:Gu,bitwiseAndImpl:lr,castImpl:ir,ceilImpl:dr,concatImpl:Xu,equalImpl:pr,expImpl:hr,expm1Impl:fr,floorDivImpl:xr,floorImpl:mr,gatherNdImpl:Zu,gatherV2Impl:Ju,greaterEqualImpl:Cr,greaterImpl:gr,lessEqualImpl:br,lessImpl:$r,linSpaceImpl:od,logImpl:vr,maxImpl:ad,maximumImpl:wr,minimumImpl:Ir,multiplyImpl:ts,negImpl:Rr,notEqualImpl:yr,prodImpl:Er,raggedGatherImpl:bd,raggedRangeImpl:vd,raggedTensorToTensorImpl:wd,rangeImpl:Id,rsqrtImpl:Nr,scatterImpl:yd,sigmoidImpl:Sd,simpleAbsImpl:rr,sliceImpl:kr,sparseFillEmptyRowsImpl:Nd,sparseReshapeImpl:kd,sparseSegmentReductionImpl:Ad,sqrtImpl:Od,squaredDifferenceImpl:Ar,staticRegexReplaceImpl:Or,stridedSliceImpl:_d,stringNGramsImpl:Vd,stringSplitImpl:Md,stringToHashBucketFastImpl:Wd,subImpl:Dr,tileImpl:zd,topKImpl:Hd,transposeImpl:Sr,uniqueImpl:Xd},Symbol.toStringTag,{value:"Module"}));const ke={},Et={alpha:!1,antialias:!1,premultipliedAlpha:!1,preserveDrawingBuffer:!1,depth:!1,stencil:!1,failIfMajorPerformanceCaveat:!0};function Pr(n,e){ke[n]=e}function ue(n,e){if(!(n in ke)||e!=null){const s=qd(n,e);if(s!==null)ke[n]=s;else return console.log("Could not get context for WebGL version",n),null}const t=ke[n];return t==null||t.isContextLost()?(delete ke[n],ue(n)):(t.disable(t.DEPTH_TEST),t.disable(t.STENCIL_TEST),t.disable(t.BLEND),t.disable(t.DITHER),t.disable(t.POLYGON_OFFSET_FILL),t.disable(t.SAMPLE_COVERAGE),t.enable(t.SCISSOR_TEST),t.enable(t.CULL_FACE),t.cullFace(t.BACK),ke[n])}function jd(n){if(!w().getBool("IS_SAFARI")&&typeof OffscreenCanvas<"u"&&n===2)return new OffscreenCanvas(300,150);if(typeof document<"u")return document.createElement("canvas");throw new Error("Cannot create a canvas in this context")}function qd(n,e){if(n!==1&&n!==2)throw new Error("Cannot get WebGL rendering context, WebGL is disabled.");const t=e??jd(n);return t.addEventListener("webglcontextlost",s=>{s.preventDefault(),delete ke[n]},!1),w().getBool("SOFTWARE_WEBGL_ENABLED")&&(Et.failIfMajorPerformanceCaveat=!1),n===1?t.getContext("webgl",Et)||t.getContext("experimental-webgl",Et):t.getContext("webgl2",Et)}var ft;(function(n){n[n.DENSE=0]="DENSE",n[n.SHARED_BATCH=1]="SHARED_BATCH"})(ft||(ft={}));var te;(function(n){n[n.RENDER=0]="RENDER",n[n.UPLOAD=1]="UPLOAD",n[n.PIXELS=2]="PIXELS",n[n.DOWNLOAD=3]="DOWNLOAD"})(te||(te={}));var L;(function(n){n[n.UNPACKED_FLOAT16=0]="UNPACKED_FLOAT16",n[n.UNPACKED_FLOAT32=1]="UNPACKED_FLOAT32",n[n.PACKED_4X1_UNSIGNED_BYTE=2]="PACKED_4X1_UNSIGNED_BYTE",n[n.PACKED_2X2_FLOAT32=3]="PACKED_2X2_FLOAT32",n[n.PACKED_2X2_FLOAT16=4]="PACKED_2X2_FLOAT16"})(L||(L={}));function wt(n,e){return[e,n]}function Yd(n,e){return n*e}function Nt(n){const e=E(n),t=Math.ceil(e/4);return an(t)}function qe(n,e){return[Math.max(1,Math.ceil(e/2)),Math.max(1,Math.ceil(n/2))]}function Qd(n,e){const[t,s]=qe(n,e);return t*s*4}function ns(n,e){const t=n;let s,o,r,a,c,i,l,u,d,p;return w().getNumber("WEBGL_VERSION")===2?(s=t.R32F,o=t.R16F,r=t.RGBA16F,a=t.RGBA32F,c=t.RED,l=4,u=1,d=t.HALF_FLOAT,p=t.FLOAT,i=t.RGBA8):(s=n.RGBA,o=n.RGBA,r=n.RGBA,a=t.RGBA,c=n.RGBA,l=4,u=4,d=e!=null?e.HALF_FLOAT_OES:null,p=n.FLOAT,i=n.RGBA),{internalFormatFloat:s,internalFormatHalfFloat:o,internalFormatPackedHalfFloat:r,internalFormatPackedFloat:a,textureFormatFloat:c,downloadTextureFormat:i,downloadUnpackNumChannels:l,defaultNumChannels:u,textureTypeHalfFloat:d,textureTypeFloat:p}}function y(n,e){const t=e();return w().getBool("DEBUG")&&Zd(n),t}function Zd(n){const e=n.getError();if(e!==n.NO_ERROR)throw new Error("WebGL Error: "+Lr(n,e))}const Jd=596e-10,ep=65504;function _r(n){return!!(w().getBool("WEBGL_RENDER_FLOAT32_ENABLED")||n===0||Jd<Math.abs(n)&&Math.abs(n)<ep)}function Lr(n,e){switch(e){case n.NO_ERROR:return"NO_ERROR";case n.INVALID_ENUM:return"INVALID_ENUM";case n.INVALID_VALUE:return"INVALID_VALUE";case n.INVALID_OPERATION:return"INVALID_OPERATION";case n.INVALID_FRAMEBUFFER_OPERATION:return"INVALID_FRAMEBUFFER_OPERATION";case n.OUT_OF_MEMORY:return"OUT_OF_MEMORY";case n.CONTEXT_LOST_WEBGL:return"CONTEXT_LOST_WEBGL";default:return`Unknown error code ${e}`}}function at(n,e){return be(n,()=>n.getExtension(e),'Extension "'+e+'" not supported on this browser.')}function Vr(n,e){const t=be(n,()=>n.createShader(n.VERTEX_SHADER),"Unable to create vertex WebGLShader.");if(y(n,()=>n.shaderSource(t,e)),y(n,()=>n.compileShader(t)),n.getShaderParameter(t,n.COMPILE_STATUS)===!1)throw console.log(n.getShaderInfoLog(t)),new Error("Failed to compile vertex shader.");return t}function Br(n,e){const t=be(n,()=>n.createShader(n.FRAGMENT_SHADER),"Unable to create fragment WebGLShader.");if(y(n,()=>n.shaderSource(t,e)),y(n,()=>n.compileShader(t)),w().get("ENGINE_COMPILE_ONLY"))return t;if(n.getShaderParameter(t,n.COMPILE_STATUS)===!1)throw ss(e,n.getShaderInfoLog(t)),new Error("Failed to compile fragment shader.");return t}const tp=/ERROR: [0-9]+:([0-9]+):/g;function ss(n,e){const t=tp.exec(e);if(t==null){console.log(`Couldn't parse line number in error: ${e}`),console.log(n);return}const s=+t[1],o=n.split(`
`),r=o.length.toString().length+2,a=o.map((d,p)=>Cs((p+1).toString(),r)+d);let c=0;for(let d=0;d<a.length;d++)c=Math.max(a[d].length,c);const i=a.slice(0,s-1),l=a.slice(s-1,s),u=a.slice(s);console.log(i.join(`
`)),console.log(e.split(`
`)[0]),console.log(`%c ${Cs(l[0],c)}`,"border:1px solid red; background-color:#e3d2d2; color:#a61717"),console.log(u.join(`
`))}function Mr(n){return be(n,()=>n.createProgram(),"Unable to create WebGLProgram.")}function Wr(n,e){if(y(n,()=>n.linkProgram(e)),!w().get("ENGINE_COMPILE_ONLY")&&n.getProgramParameter(e,n.LINK_STATUS)===!1)throw console.log(n.getProgramInfoLog(e)),new Error("Failed to link vertex and fragment shaders.")}function Dt(n,e){if(y(n,()=>n.validateProgram(e)),n.getProgramParameter(e,n.VALIDATE_STATUS)===!1)throw console.log(n.getProgramInfoLog(e)),new Error("Shader program validation failed.")}function Ur(n,e){const t=be(n,()=>n.createBuffer(),"Unable to create WebGLBuffer");return y(n,()=>n.bindBuffer(n.ARRAY_BUFFER,t)),y(n,()=>n.bufferData(n.ARRAY_BUFFER,e,n.STATIC_DRAW)),t}function Gr(n,e){const t=be(n,()=>n.createBuffer(),"Unable to create WebGLBuffer");return y(n,()=>n.bindBuffer(n.ELEMENT_ARRAY_BUFFER,t)),y(n,()=>n.bufferData(n.ELEMENT_ARRAY_BUFFER,e,n.STATIC_DRAW)),t}function np(){return w().getNumber("WEBGL_VERSION")===2?1:4}function zr(n){return be(n,()=>n.createTexture(),"Unable to create WebGLTexture.")}function Hr(n,e){const t=w().getNumber("WEBGL_MAX_TEXTURE_SIZE");if(n<=0||e<=0){const s=`[${n}x${e}]`;throw new Error("Requested texture size "+s+" is invalid.")}if(n>t||e>t){const s=`[${n}x${e}]`,o=`[${t}x${t}]`;throw new Error("Requested texture size "+s+" greater than WebGL maximum on this browser / GPU "+o+".")}}function Xr(n){return be(n,()=>n.createFramebuffer(),"Unable to create WebGLFramebuffer.")}function pn(n,e,t,s,o,r,a){const c=n.getAttribLocation(e,t);return c===-1?!1:(y(n,()=>n.bindBuffer(n.ARRAY_BUFFER,s)),y(n,()=>n.vertexAttribPointer(c,o,n.FLOAT,!1,r,a)),y(n,()=>n.enableVertexAttribArray(c)),!0)}function Kr(n,e,t){Zr(n,t),y(n,()=>n.activeTexture(n.TEXTURE0+t)),y(n,()=>n.bindTexture(n.TEXTURE_2D,e))}function sp(n,e){Zr(n,e),y(n,()=>n.activeTexture(n.TEXTURE0+e)),y(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function jr(n,e,t){return be(n,()=>n.getUniformLocation(e,t),'uniform "'+t+'" not present in program.')}function qr(n,e,t){return n.getUniformLocation(e,t)}function Yr(n,e,t,s){y(n,()=>Kr(n,e,s)),y(n,()=>n.uniform1i(t,s))}function op(n){y(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,null)),y(n,()=>n.viewport(0,0,n.canvas.width,n.canvas.height)),y(n,()=>n.scissor(0,0,n.canvas.width,n.canvas.height))}function Ft(n,e,t){y(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,t)),y(n,()=>n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,e,0))}function hn(n,e){y(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,e)),y(n,()=>n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,null,0))}function it(n){const e=n.checkFramebufferStatus(n.FRAMEBUFFER);if(e!==n.FRAMEBUFFER_COMPLETE)throw new Error("Error binding framebuffer: "+Qr(n,e))}function Qr(n,e){switch(e){case n.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_ATTACHMENT";case n.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";case n.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:return"FRAMEBUFFER_INCOMPLETE_DIMENSIONS";case n.FRAMEBUFFER_UNSUPPORTED:return"FRAMEBUFFER_UNSUPPORTED";default:return`unknown error ${e}`}}function be(n,e,t){const s=y(n,()=>e());if(s==null)throw new Error(t);return s}function Zr(n,e){const t=n.MAX_COMBINED_TEXTURE_IMAGE_UNITS-1,s=e+n.TEXTURE0;if(s<n.TEXTURE0||s>t){const o=`[gl.TEXTURE0, gl.TEXTURE${t}]`;throw new Error(`textureUnit must be in ${o}.`)}}function Oe(n,e=2){return E(n.slice(0,n.length-e))}function De(n){if(n.length===0)throw Error("Cannot get rows and columns of an empty shape array.");return[n.length>1?n[n.length-2]:1,n[n.length-1]]}function ct(n){let e=[1,1,1];return n.length===0||n.length===1&&n[0]===1||(e=[Oe(n),...De(n)]),e}function Jr(n,e=!1){let t=w().getNumber("WEBGL_MAX_TEXTURE_SIZE"),s=w().getNumber("WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE");s===1/0&&w().getBool("WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE")&&(s=t/2),e&&(t=t*2,s=s*2,n=n.map((c,i)=>i>=n.length-2?Gn(n[i]):n[i]),n.length===1&&(n=[2,n[0]])),n.length!==2&&(n=Pe(n).newShape);let o=E(n),r=null;n.length<=1&&o<=t?r=[1,o]:n.length===2&&n[0]<=t&&n[1]<=t?r=n:n.length===3&&n[0]*n[1]<=t&&n[2]<=t?r=[n[0]*n[1],n[2]]:n.length===3&&n[0]<=t&&n[1]*n[2]<=t?r=[n[0],n[1]*n[2]]:n.length===4&&n[0]*n[1]*n[2]<=t&&n[3]<=t?r=[n[0]*n[1]*n[2],n[3]]:n.length===4&&n[0]<=t&&n[1]*n[2]*n[3]<=t&&(r=[n[0],n[1]*n[2]*n[3]]);const a=r!=null&&Math.max(...r)>s&&Math.min(...r)<=(e?2:1)&&Math.min(...r)>0;if(r==null||a)if(e){const c=Oe(n);let i=2,l=2;n.length&&([i,l]=De(n)),o=c*(i/2)*(l/2),r=an(o).map(u=>u*2)}else r=an(o);return r}function kt(n){return n%2===0}function mt(n,e){if(n=n.slice(-2),e=e.slice(-2),q(n,e)||!n.length||!e.length||n[0]===0||n[1]===0||e[0]===0||e[1]===0)return!0;if(n.length!==e.length){const t=n[n.length-1],s=e[e.length-1];if(t===s||kt(t)&&kt(s)&&(n[0]===1||e[0]===1))return!0}return n[1]===e[1]&&kt(n[0])&&kt(e[0])}let Pt,_t;function ea(n){if(Pt==null){const e=ue(n);Pt=e.getParameter(e.MAX_TEXTURE_SIZE)}return Pt}function rp(){Pt=null}function ap(){_t=null}function ta(n){if(_t==null){const e=ue(n);_t=e.getParameter(e.MAX_TEXTURE_IMAGE_UNITS)}return Math.min(16,_t)}function na(n){if(n===0)return 0;let e;const t=ue(n);return ne(t,"EXT_disjoint_timer_query_webgl2")&&n===2?e=2:ne(t,"EXT_disjoint_timer_query")?e=1:e=0,e}function ne(n,e){return n.getExtension(e)!=null}function fn(n){try{if(ue(n)!=null)return!0}catch(e){return console.log("Error when getting WebGL context: ",e),!1}return!1}function sa(n){if(n===0)return!1;const e=ue(n);if(n===1){if(!ne(e,"OES_texture_float"))return!1}else if(!ne(e,"EXT_color_buffer_float"))return!1;return mn(e)}function oa(n){if(n===0)return!1;const e=ue(n);if(n===1){if(!ne(e,"OES_texture_float")||!ne(e,"WEBGL_color_buffer_float"))return!1}else{if(ne(e,"EXT_color_buffer_float"))return mn(e);const s="EXT_color_buffer_half_float";if(ne(e,s)){const o=e.getExtension(s);return ip(e,o)}return!1}return mn(e)}function mn(n){const e=ns(n),t=n.createTexture();n.bindTexture(n.TEXTURE_2D,t),n.texImage2D(n.TEXTURE_2D,0,e.internalFormatFloat,1,1,0,e.textureFormatFloat,e.textureTypeFloat,null);const r=n.createFramebuffer();n.bindFramebuffer(n.FRAMEBUFFER,r),n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,t,0);const a=n.checkFramebufferStatus(n.FRAMEBUFFER)===n.FRAMEBUFFER_COMPLETE;return n.bindTexture(n.TEXTURE_2D,null),n.bindFramebuffer(n.FRAMEBUFFER,null),n.deleteTexture(t),n.deleteFramebuffer(r),a}function ip(n,e){const t=ns(n,e),s=n.createTexture();n.bindTexture(n.TEXTURE_2D,s),n.texImage2D(n.TEXTURE_2D,0,t.internalFormatHalfFloat,1,1,0,t.textureFormatFloat,t.textureTypeHalfFloat,null);const a=n.createFramebuffer();n.bindFramebuffer(n.FRAMEBUFFER,a),n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,s,0);const c=n.checkFramebufferStatus(n.FRAMEBUFFER)===n.FRAMEBUFFER_COMPLETE;return n.bindTexture(n.TEXTURE_2D,null),n.bindFramebuffer(n.FRAMEBUFFER,null),n.deleteTexture(s),n.deleteFramebuffer(a),c}function ra(n){return n!==2?!1:ue(n).fenceSync!=null}function Ye(n,e){Array.isArray(n)||(n=[n]),n.forEach(t=>{t!=null&&O(t.dtype!=="complex64",()=>`${e} does not support complex64 tensors in the WebGL backend.`)})}const cp=Object.freeze(Object.defineProperty({__proto__:null,assertNotComplex:Ye,bindCanvasToFramebuffer:op,bindColorTextureToFramebuffer:Ft,bindTextureToProgramUniformSampler:Yr,bindTextureUnit:Kr,bindVertexBufferToProgramAttribute:pn,callAndCheck:y,canBeRepresented:_r,createFragmentShader:Br,createFramebuffer:Xr,createProgram:Mr,createStaticIndexBuffer:Gr,createStaticVertexBuffer:Ur,createTexture:zr,createVertexShader:Vr,getBatchDim:Oe,getExtensionOrThrow:at,getFramebufferErrorMessage:Qr,getMaxTexturesInShader:ta,getNumChannels:np,getProgramUniformLocation:qr,getProgramUniformLocationOrThrow:jr,getRowsCols:De,getShapeAs3D:ct,getTextureShapeFromLogicalShape:Jr,getWebGLDisjointQueryTimerVersion:na,getWebGLErrorMessage:Lr,getWebGLMaxTextureSize:ea,hasExtension:ne,isCapableOfRenderingToFloatTexture:sa,isDownloadFloatTextureEnabled:oa,isReshapeFree:mt,isWebGLFenceEnabled:ra,isWebGLVersionEnabled:fn,linkProgram:Wr,logShaderSourceAndInfoLog:ss,resetMaxTextureSize:rp,resetMaxTexturesInShader:ap,unbindColorTextureFromFramebuffer:hn,unbindTextureUnit:sp,validateFramebuffer:it,validateProgram:Dt,validateTextureSize:Hr},Symbol.toStringTag,{value:"Module"}));const N=w();N.registerFlag("HAS_WEBGL",()=>N.getNumber("WEBGL_VERSION")>0);N.registerFlag("WEBGL_VERSION",()=>fn(2)?2:fn(1)?1:0);N.registerFlag("WEBGL_CHECK_NUMERICAL_PROBLEMS",()=>!1);N.registerFlag("WEBGL_BUFFER_SUPPORTED",()=>N.get("WEBGL_VERSION")===2);N.registerFlag("WEBGL_CPU_FORWARD",()=>!0);N.registerFlag("WEBGL_FORCE_F16_TEXTURES",()=>!1);N.registerFlag("WEBGL_PACK",()=>N.getBool("HAS_WEBGL"));N.registerFlag("WEBGL_PACK_NORMALIZATION",()=>N.getBool("WEBGL_PACK"));N.registerFlag("WEBGL_PACK_CLIP",()=>N.getBool("WEBGL_PACK"));N.registerFlag("WEBGL_PACK_DEPTHWISECONV",()=>N.getBool("WEBGL_PACK"));N.registerFlag("WEBGL_PACK_BINARY_OPERATIONS",()=>N.getBool("WEBGL_PACK"));N.registerFlag("WEBGL_PACK_UNARY_OPERATIONS",()=>N.getBool("WEBGL_PACK"));N.registerFlag("WEBGL_PACK_ARRAY_OPERATIONS",()=>N.getBool("WEBGL_PACK"));N.registerFlag("WEBGL_PACK_IMAGE_OPERATIONS",()=>N.getBool("WEBGL_PACK"));N.registerFlag("WEBGL_PACK_REDUCE",()=>N.getBool("WEBGL_PACK"));N.registerFlag("WEBGL_LAZILY_UNPACK",()=>N.getBool("WEBGL_PACK"));N.registerFlag("WEBGL_CONV_IM2COL",()=>N.getBool("WEBGL_PACK"));N.registerFlag("WEBGL_PACK_CONV2DTRANSPOSE",()=>N.getBool("WEBGL_PACK"));N.registerFlag("WEBGL_MAX_TEXTURE_SIZE",()=>ea(N.getNumber("WEBGL_VERSION")));N.registerFlag("WEBGL_MAX_TEXTURES_IN_SHADER",()=>ta(N.getNumber("WEBGL_VERSION")));N.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION",()=>{const n=N.getNumber("WEBGL_VERSION");return n===0?0:na(n)});N.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE",()=>N.getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0&&!uo());N.registerFlag("WEBGL_RENDER_FLOAT32_CAPABLE",()=>sa(N.getNumber("WEBGL_VERSION")));N.registerFlag("WEBGL_RENDER_FLOAT32_ENABLED",()=>N.getBool("WEBGL_FORCE_F16_TEXTURES")?!1:N.getBool("WEBGL_RENDER_FLOAT32_CAPABLE"));N.registerFlag("WEBGL_DOWNLOAD_FLOAT_ENABLED",()=>oa(N.getNumber("WEBGL_VERSION")));N.registerFlag("WEBGL_FENCE_API_ENABLED",()=>ra(N.getNumber("WEBGL_VERSION")));N.registerFlag("WEBGL_SIZE_UPLOAD_UNIFORM",()=>N.getBool("WEBGL_RENDER_FLOAT32_ENABLED")?4:0);N.registerFlag("WEBGL_DELETE_TEXTURE_THRESHOLD",()=>-1,n=>{if(typeof n!="number")throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be a number but got ${n}.`);if(n<0&&n!==-1)throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be -1 (indicating never delete) or at least 0, but got ${n}.`)});N.registerFlag("WEBGL_FLUSH_THRESHOLD",()=>uo()?1:-1,n=>{if(typeof n!="number")throw new Error(`WEBGL_FLUSH_THRESHOLD must be a number but got ${n}.`);if(n<0&&n!==-1)throw new Error(`WEBGL_FLUSH_THRESHOLD must be -1 (indicating never manual flush) or at least 0, but got ${n}.`)});N.registerFlag("CPU_HANDOFF_SIZE_THRESHOLD",()=>128);N.registerFlag("WEBGL_USE_SHAPES_UNIFORMS",()=>!1);N.registerFlag("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD",()=>1e5);N.registerFlag("TOPK_K_CPU_HANDOFF_THRESHOLD",()=>128);N.registerFlag("WEBGL_EXP_CONV",()=>!1);N.registerFlag("SOFTWARE_WEBGL_ENABLED",()=>N.getBool("IS_TEST"));N.registerFlag("WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE",()=>1/0);N.registerFlag("WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE",()=>!1);N.registerFlag("WEBGL2_ISNAN_CUSTOM",()=>!1);N.registerFlag("ENGINE_COMPILE_ONLY",()=>!1);function K(){let n,e,t,s,o,r,a,c,i,l;return w().getNumber("WEBGL_VERSION")===2?(n="#version 300 es",e="in",t="out",s="in",o="texture",r="outputColor",a="out vec4 outputColor;",c=w().getBool("WEBGL2_ISNAN_CUSTOM")?`
      bool isnan_custom(float val) {
        uint floatToUint = floatBitsToUint(val);
        return (floatToUint & 0x7fffffffu) > 0x7f800000u;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `:"",i="",l=`
      #define round(value) newRound(value)
      int newRound(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 newRound(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `):(n="",e="attribute",t="varying",s="varying",o="texture2D",r="gl_FragColor",a="",c=`
      #define isnan(value) isnan_custom(value)
      bool isnan_custom(float val) {
        return (val > 0. || val < 1. || val == 0.) ? false : true;
      }
      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));
      }
    `,i=`
      uniform float INFINITY;

      bool isinf(float val) {
        return abs(val) == INFINITY;
      }
      bvec4 isinf(vec4 val) {
        return equal(abs(val), vec4(INFINITY));
      }
    `,l=`
      int round(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 round(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `),{version:n,attribute:e,varyingVs:t,varyingFs:s,texture2D:o,output:r,defineOutput:a,defineSpecialNaN:c,defineSpecialInf:i,defineRound:l}}function Le(n,e,t="index"){const s=M(e);return s.map((o,r)=>{const a=`int ${n[r]} = ${t} / ${o}`,c=r===s.length-1?`int ${n[r+1]} = ${t} - ${n[r]} * ${o}`:`index -= ${n[r]} * ${o}`;return`${a}; ${c};`}).join("")}function Yt(n,e,t="index"){const s=M(e);return s.map((o,r)=>{const a=`int ${n[r]} = ${t} / outShapeStrides[${r}]`,c=r===s.length-1?`int ${n[r+1]} = ${t} - ${n[r]} * outShapeStrides[${r}]`:`index -= ${n[r]} * outShapeStrides[${r}]`;return`${a}; ${c};`}).join("")}function lp(n,e){const t=n.length,s=n.map(r=>`${e}[${r}]`),o=new Array(t-1);o[t-2]=s[t-1];for(let r=t-3;r>=0;--r)o[r]=`(${o[r+1]} * ${s[r+1]})`;return o}function up(n,e,t="index"){const s=n.map((r,a)=>a),o=lp(s,e);return o.map((r,a)=>{const c=`int ${n[a]} = ${t} / ${o[a]}`,i=a===o.length-1?`int ${n[a+1]} = ${t} - ${n[a]} * ${o[a]}`:`index -= ${n[a]} * ${o[a]}`;return`${c}; ${i};`}).join("")}function os(n){const e=M(n).map(t=>t.toString());return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * ${e[0]} + coords.y * ${e[1]} + coords.z;
  }
`}function rs(){return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * outShapeStrides[0] + coords.y * outShapeStrides[1] + coords.z;
  }
`}const aa=`
  const float FLOAT_MAX = 1.70141184e38;
  const float FLOAT_MIN = 1.17549435e-38;

  lowp vec4 encode_float(highp float v) {
    if (isnan(v)) {
      return vec4(255, 255, 255, 255);
    }

    highp float av = abs(v);

    if(av < FLOAT_MIN) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    } else if(v > FLOAT_MAX) {
      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;
    } else if(v < -FLOAT_MAX) {
      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;
    }

    highp vec4 c = vec4(0,0,0,0);

    highp float e = floor(log2(av));
    highp float m = exp2(fract(log2(av))) - 1.0;

    c[2] = floor(128.0 * m);
    m -= c[2] / 128.0;
    c[1] = floor(32768.0 * m);
    m -= c[1] / 32768.0;
    c[0] = floor(8388608.0 * m);

    highp float ebias = e + 127.0;
    c[3] = floor(ebias / 2.0);
    ebias -= c[3] * 2.0;
    c[2] += floor(ebias) * 128.0;

    c[3] += 128.0 * step(0.0, -v);

    return c / 255.0;
  }
`;const{getBroadcastDims:ia}=Vu;function dp(n,e,t){const s=[];if(n.forEach(h=>{const f=E(h.shapeInfo.logicalShape);if(h.shapeInfo.isUniform?s.push(`uniform float ${h.name}${f>1?`[${f}]`:""};`):(s.push(`uniform sampler2D ${h.name};`),s.push(`uniform int offset${h.name};`)),t.enableShapeUniforms){const{uniformShape:g}=as(t.packedInputs,h.shapeInfo.logicalShape,h.shapeInfo.texShape);switch(g.length){case 1:s.push(`uniform int ${h.name}Shape;`);break;case 2:s.push(`uniform ivec2 ${h.name}Shape;`);break;case 3:s.push(`uniform ivec3 ${h.name}Shape;`);break;case 4:s.push(`uniform ivec4 ${h.name}Shape;`);break}s.push(`uniform ivec2 ${h.name}TexShape;`)}}),t.enableShapeUniforms){switch(e.logicalShape.length){case 1:s.push("uniform int outShape;");break;case 2:s.push("uniform ivec2 outShape;"),s.push("uniform int outShapeStrides;");break;case 3:s.push("uniform ivec3 outShape;"),s.push("uniform ivec2 outShapeStrides;");break;case 4:s.push("uniform ivec4 outShape;"),s.push("uniform ivec3 outShapeStrides;");break}s.push("uniform ivec2 outTexShape;")}t.customUniforms&&t.customUniforms.forEach(h=>{s.push(`uniform ${h.type} ${h.name}${h.arrayIndex?`[${h.arrayIndex}]`:""};`)});const o=s.join(`
`),r=n.map(h=>pp(h,e,t.packedInputs,t.enableShapeUniforms)).join(`
`),a=e.texShape,c=K(),i=mp(c);let l,u,d=Cp(c);return e.isPacked?(l=hp(e.logicalShape,a,t.enableShapeUniforms),u=gp(c)):(l=fp(e.logicalShape,a,t.enableShapeUniforms),u=xp(c)),t.packedInputs&&(d+=wp),[d,i,u,o,l,r,t.userCode].join(`
`)}function Qe(n,e=!1){const t=n.shapeInfo.logicalShape;switch(t.length){case 0:return Fp(n,e);case 1:return _p(n,e);case 2:return Vp(n,e);case 3:return Mp(n,e);case 4:return Up(n,e);case 5:return Gp(n);case 6:return zp(n);default:throw new Error(`${t.length}-D input sampling is not yet supported`)}}function ca(n,e){switch(n.shapeInfo.logicalShape.length){case 0:return Dp(n);case 1:return Pp(n,e);case 2:return Lp(n,e);case 3:return Bp(n,e);default:return Wp(n,e)}}function pp(n,e,t=!1,s){let o="";t?o+=ca(n,s):o+=Qe(n,s);const r=n.shapeInfo.logicalShape,a=e.logicalShape;return r.length<=a.length&&(t?o+=Hp(n,e):o+=Xp(n,e)),o}function hp(n,e,t){switch(n.length){case 0:return la();case 1:return Ip(n,e,t);case 2:return Ap(n,e,t);case 3:return yp(n,e,t);default:return Tp(n,e,t)}}function fp(n,e,t){switch(n.length){case 0:return la();case 1:return Rp(n,e,t);case 2:return Op(n,e,t);case 3:return Sp(n,e,t);case 4:return Ep(n,e,t);case 5:return Np(n,e);case 6:return kp(n,e);default:throw new Error(`${n.length}-D output sampling is not yet supported`)}}function mp(n){return`
    float sampleTexture(sampler2D textureSampler, vec2 uv) {
      return ${n.texture2D}(textureSampler, uv).r;
    }
  `}function xp(n){return`
    void setOutput(float val) {
      ${n.output} = vec4(val, 0, 0, 0);
    }
  `}function gp(n){return`
    void setOutput(vec4 val) {
      ${n.output} = val;
    }
  `}function Cp(n){return`${n.version}
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    ${n.varyingFs} vec2 resultUV;
    ${n.defineOutput}
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    uniform float NAN;
    ${n.defineSpecialNaN}
    ${n.defineSpecialInf}
    ${n.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    int idiv(int a, int b, float sign) {
      int res = a / b;
      int mod = imod(a, b);
      if (sign < 0. && mod != 0) {
        res -= 1;
      }
      return res;
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975
    float random(float seed){
      vec2 p = resultUV * seed;
      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
      p3 += dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${$p}
    ${bp}
    ${vp}
  `}const $p=`
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,bp=`
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,vp=`
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,wp=`
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;function la(){return`
    int getOutputCoords() {
      return 0;
    }
  `}function Ip(n,e,t){const s=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)];return s[0]===1?t?`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ceil(float(outTexShape[1]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${s[1]}.0);
      }
    `:s[1]===1?t?`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ceil(float(outTexShape[0]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${s[0]}.0);
      }
    `:t?`
    int getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      return 2 * (resTexRC.x * packedTexShape[1] + resTexRC.y);
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      return 2 * (resTexRC.x * ${s[1]} + resTexRC.y);
    }
  `}function Rp(n,e,t){return e[0]===1?t?`
      int getOutputCoords() {
        return int(resultUV.x * float(outTexShape[1]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.x * ${e[1]}.0);
      }
    `:e[1]===1?t?`
      int getOutputCoords() {
        return int(resultUV.y * float(outTexShape[0]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.y * ${e[0]}.0);
      }
    `:t?`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      return resTexRC.x * outTexShape[1] + resTexRC.y;
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${e[0]}, ${e[1]}));
      return resTexRC.x * ${e[1]} + resTexRC.y;
    }
  `}function yp(n,e,t){if(t)return`
    ivec3 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec3(b, r, c);
    }
  `;const s=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)],o=Math.ceil(n[2]/2),r=o*Math.ceil(n[1]/2);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      int index = resTexRC.x * ${s[1]} + resTexRC.y;

      int b = index / ${r};
      index -= b * ${r};

      int r = 2 * (index / ${o});
      int c = imod(index, ${o}) * 2;

      return ivec3(b, r, c);
    }
  `}function Sp(n,e,t){if(t)return`
  ivec3 getOutputCoords() {
    ivec2 resTexRC = ivec2(resultUV.yx *
                           vec2(outTexShape[0], outTexShape[1]));
    int index = resTexRC.x * outTexShape[1] + resTexRC.y;
    ${Yt(["r","c","d"],n)}
    return ivec3(r, c, d);
  }
`;const s=Le(["r","c","d"],n);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${e[0]}, ${e[1]}));
      int index = resTexRC.x * ${e[1]} + resTexRC.y;
      ${s}
      return ivec3(r, c, d);
    }
  `}function Tp(n,e,t){if(t)return`
    ivec4 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int texelsInLogicalRow = int(ceil(float(outShape[3]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatchN = texelsInBatch * outShape[1];

      int b2 = index / texelsInBatchN;
      index -= b2 * texelsInBatchN;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec4(b2, b, r, c);
    }
  `;const s=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)],o=Math.ceil(n[n.length-1]/2),r=o*Math.ceil(n[n.length-2]/2);let a=r,c="",i="b, r, c";for(let l=2;l<n.length-1;l++)a*=n[n.length-l-1],c=`
      int b${l} = index / ${a};
      index -= b${l} * ${a};
    `+c,i=`b${l}, `+i;return`
    ivec${n.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      int index = resTexRC.x * ${s[1]} + resTexRC.y;

      ${c}

      int b = index / ${r};
      index -= b * ${r};

      int r = 2 * (index / ${o});
      int c = imod(index, ${o}) * 2;

      return ivec${n.length}(${i});
    }
  `}function Ep(n,e,t){if(t)return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      ${Yt(["r","c","d","d2"],n)}
      return ivec4(r, c, d, d2);
    }
  `;const s=Le(["r","c","d","d2"],n);return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${e[0]}, ${e[1]}));
      int index = resTexRC.x * ${e[1]} + resTexRC.y;
      ${s}
      return ivec4(r, c, d, d2);
    }
  `}function Np(n,e){const t=Le(["r","c","d","d2","d3"],n);return`
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${e[0]},
                             ${e[1]}));

      int index = resTexRC.x * ${e[1]} + resTexRC.y;

      ${t}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `}function kp(n,e){const t=Le(["r","c","d","d2","d3","d4"],n);return`
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${e[0]}, ${e[1]}));
      int index = resTexRC.x * ${e[1]} + resTexRC.y;

      ${t}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `}function Ap(n,e,t){const s=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)];if(q(n,e))return t?`
      ivec2 getOutputCoords() {
        ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
        return 2 * ivec2(resultUV.yx * vec2(packedTexShape[0], packedTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${s[0]}, ${s[1]}));
      }
    `;const o=Math.ceil(n[1]/2);return t?`
    ivec2 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));

      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;
      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));

      int index = resTexRC.x * ${s[1]} + resTexRC.y;
      int r = 2 * (index / ${o});
      int c = imod(index, ${o}) * 2;

      return ivec2(r, c);
    }
  `}function Op(n,e,t){return q(n,e)?t?`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(outTexShape[0], outTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${e[0]}, ${e[1]}));
      }
    `:n[1]===1?t?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(index, 0);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${e[0]}, ${e[1]}));
        int index = resTexRC.x * ${e[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `:n[0]===1?t?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(0, index);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${e[0]}, ${e[1]}));
        int index = resTexRC.x * ${e[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `:t?`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      int r = index / outShape[1];
      int c = index - r * outShape[1];
      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${e[0]}, ${e[1]}));
      int index = resTexRC.x * ${e[1]} + resTexRC.y;
      int r = index / ${n[1]};
      int c = index - r * ${n[1]};
      return ivec2(r, c);
    }
  `}function Ve(n){return`offset${n}`}function Dp(n){const e=n.name,t="get"+e.charAt(0).toUpperCase()+e.slice(1),s=K();return`
    vec4 ${t}() {
      return ${s.texture2D}(${e}, halfCR);
    }
  `}function Fp(n,e){const t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1);if(n.shapeInfo.isUniform)return`float ${s}() {return ${t};}`;const[o,r]=n.shapeInfo.texShape;if(o===1&&r===1)return`
      float ${s}() {
        return sampleTexture(${t}, halfCR);
      }
    `;const a=Ve(t);if(e)return`
    float ${s}() {
      vec2 uv = uvFromFlat(${t}TexShape[0], ${t}TexShape[1], ${a});
      return sampleTexture(${t}, uv);
    }
  `;const[c,i]=n.shapeInfo.texShape;return`
    float ${s}() {
      vec2 uv = uvFromFlat(${c}, ${i}, ${a});
      return sampleTexture(${t}, uv);
    }
  `}function Pp(n,e){const t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1),o=n.shapeInfo.texShape,r=K();if(e)return`
    vec4 ${s}(int index) {
      ivec2 packedTexShape = ivec2(ceil(float(${t}TexShape[0]) / 2.0), ceil(float(${t}TexShape[1]) / 2.0));
      vec2 uv = packedUVfrom1D(
        packedTexShape[0], packedTexShape[1], index);
      return ${r.texture2D}(${t}, uv);
    }
  `;const a=[Math.ceil(o[0]/2),Math.ceil(o[1]/2)];return`
    vec4 ${s}(int index) {
      vec2 uv = packedUVfrom1D(
        ${a[0]}, ${a[1]}, index);
      return ${r.texture2D}(${t}, uv);
    }
  `}function _p(n,e){const t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1);if(n.shapeInfo.isUniform)return`
      float ${s}(int index) {
        ${Ze(n)}
      }
    `;const o=n.shapeInfo.texShape,r=o[0],a=o[1];if(a===1&&r===1)return`
      float ${s}(int index) {
        return sampleTexture(${t}, halfCR);
      }
    `;const c=Ve(t);return a===1?e?`
      float ${s}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${c}) + 0.5) / float(${t}TexShape[0]));
        return sampleTexture(${t}, uv);
      }
    `:`
      float ${s}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${c}) + 0.5) / ${r}.0);
        return sampleTexture(${t}, uv);
      }
    `:r===1?e?`
      float ${s}(int index) {
        vec2 uv = vec2((float(index + ${c}) + 0.5) / float(${t}TexShape[1]), 0.5);
        return sampleTexture(${t}, uv);
      }
    `:`
      float ${s}(int index) {
        vec2 uv = vec2((float(index + ${c}) + 0.5) / ${a}.0, 0.5);
        return sampleTexture(${t}, uv);
      }
    `:e?`
    float ${s}(int index) {
      vec2 uv = uvFromFlat(${t}TexShape[0], ${t}TexShape[1], index + ${c});
      return sampleTexture(${t}, uv);
    }
  `:`
    float ${s}(int index) {
      vec2 uv = uvFromFlat(${r}, ${a}, index + ${c});
      return sampleTexture(${t}, uv);
    }
  `}function Lp(n,e){const t=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=n.shapeInfo.texShape,a=r[0],c=r[1],i=K();if(r!=null&&q(t,r))return e?`
      vec4 ${o}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);

        return ${i.texture2D}(${s}, uv);
      }
    `:`
      vec4 ${o}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${c}.0, ${a}.0);

        return ${i.texture2D}(${s}, uv);
      }
    `;if(e)return`
    vec4 ${o}(int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${s}TexShape[0]) / 2.0), ceil(float(${s}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${s}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom2D(valuesPerRow, packedTexShape[0], packedTexShape[1], row, col);
      return ${i.texture2D}(${s}, uv);
    }
  `;const l=[Math.ceil(r[0]/2),Math.ceil(r[1]/2)],u=Math.ceil(t[1]/2);return`
    vec4 ${o}(int row, int col) {
      vec2 uv = packedUVfrom2D(${u}, ${l[0]}, ${l[1]}, row, col);
      return ${i.texture2D}(${s}, uv);
    }
  `}function Vp(n,e){const t=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=n.shapeInfo.texShape;if(r!=null&&q(t,r)){if(e)return`
      float ${o}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `;const p=r[0],h=r[1];return`
    float ${o}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${h}.0, ${p}.0);
      return sampleTexture(${s}, uv);
    }
  `}const{newShape:a,keptDims:c}=Pe(t),i=a;if(i.length<t.length){const p=Je(n,i),h=["row","col"];return`
      ${Qe(p,e)}
      float ${o}(int row, int col) {
        return ${o}(${et(h,c)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${o}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${t[1]}, 1)));
        ${Ze(n)}
      }
    `;const l=r[0],u=r[1],d=Ve(s);return u===1?e?`
      float ${o}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${s}Shape[1], 1, 1));
        vec2 uv = vec2(0.5, (index + 0.5) / float(${s}TexShape[0]));
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${o}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${t[1]}, 1, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${l}.0);
      return sampleTexture(${s}, uv);
    }
  `:l===1?e?`
      float ${o}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${s}Shape[1], 1, 1));
        vec2 uv = vec2((index + 0.5) / float(${s}TexShape[1]), 0.5);
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${o}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${t[1]}, 1, 1));
      vec2 uv = vec2((index + 0.5) / ${u}.0, 0.5);
      return sampleTexture(${s}, uv);
    }
  `:e?`
      float ${o}(int row, int col) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${s}Shape[1] + col + ${d};
        vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index);
        return sampleTexture(${s}, uv);
      }
    `:`
  float ${o}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${t[1]} + col + ${d};
    vec2 uv = uvFromFlat(${l}, ${u}, index);
    return sampleTexture(${s}, uv);
  }
`}function Bp(n,e){const t=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=n.shapeInfo.texShape,a=[Math.ceil(r[0]/2),Math.ceil(r[1]/2)];if(t[0]===1){const p=t.slice(1),h=[1,2],f=Je(n,p),g=["b","row","col"];return`
        ${ca(f,e)}
        vec4 ${o}(int b, int row, int col) {
          return ${o}(${et(g,h)});
        }
      `}const c=K();if(e)return`
    vec4 ${o}(int b, int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${s}TexShape[0]) / 2.0), ceil(float(${s}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${s}Shape[2]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${s}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom3D(
        packedTexShape[0], packedTexShape[1], texelsInBatch, valuesPerRow, b, row, col);
      return ${c.texture2D}(${s}, uv);
    }
  `;const i=a[0],l=a[1],u=Math.ceil(t[2]/2),d=u*Math.ceil(t[1]/2);return`
    vec4 ${o}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${i}, ${l}, ${d}, ${u}, b, row, col);
      return ${c.texture2D}(${s}, uv);
    }
  `}function Mp(n,e){const t=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=t[1]*t[2],a=t[2],{newShape:c,keptDims:i}=Pe(t),l=c;if(l.length<t.length){const g=Je(n,l),x=["row","col","depth"];return`
        ${Qe(g,e)}
        float ${o}(int row, int col, int depth) {
          return ${o}(${et(x,i)});
        }
      `}if(n.shapeInfo.isUniform)return`
      float ${o}(int row, int col, int depth) {
        int index = round(dot(vec3(row, col, depth),
                          vec3(${r}, ${a}, 1)));
        ${Ze(n)}
      }
    `;const u=n.shapeInfo.texShape,d=u[0],p=u[1],h=n.shapeInfo.flatOffset;if(p===r&&h==null)return e?`
      float ${o}(int row, int col, int depth) {
        int stride1 = ${s}Shape[2];
        float texR = float(row);
        float texC = dot(vec2(col, depth), vec2(stride1, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
        float ${o}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${a}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${p}.0, ${d}.0);
          return sampleTexture(${s}, uv);
        }
      `;if(p===a&&h==null)return e?`
      float ${o}(int row, int col, int depth) {
        float texR = dot(vec2(row, col), vec2(${s}Shape[1], 1));
        float texC = float(depth);
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${o}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${t[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${p}.0, ${d}.0);
      return sampleTexture(${s}, uv);
    }
  `;const f=Ve(s);return e?`
    float ${o}(int row, int col, int depth) {
      // Explicitly use integer operations as dot() only works on floats.
      int stride0 = ${s}Shape[1] * ${s}Shape[2];
      int stride1 = ${s}Shape[2];
      int index = row * stride0 + col * stride1 + depth + ${f};
      vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index);
      return sampleTexture(${s}, uv);
    }
    `:`
      float ${o}(int row, int col, int depth) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${r} + col * ${a} + depth + ${f};
        vec2 uv = uvFromFlat(${d}, ${p}, index);
        return sampleTexture(${s}, uv);
      }
  `}function Wp(n,e){const t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1),o=K();if(e)return`
    vec4 ${s}(int b2, int b, int row, int col) {
      int valuesPerRow = int(ceil(float(${t}Shape[3]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${t}Shape[2]) / 2.0));
      int index = b * texelsInBatch + (row / 2) * valuesPerRow + (col / 2);
      texelsInBatch *= ${t}Shape[1];
      index = b2 * texelsInBatch + index;
      ivec2 packedTexShape = ivec2(ceil(float(${t}TexShape[0]) / 2.0), ceil(float(${t}TexShape[1]) / 2.0));
      int texR = index / packedTexShape[1];
      int texC = index - texR * packedTexShape[1];
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(packedTexShape[1], packedTexShape[0]); return ${o.texture2D}(${t}, uv);
    }
  `;const r=n.shapeInfo.logicalShape,a=r.length,c=n.shapeInfo.texShape,i=[Math.ceil(c[0]/2),Math.ceil(c[1]/2)],l=i[0],u=i[1],d=Math.ceil(r[a-1]/2);let p=d*Math.ceil(r[a-2]/2),h="int b, int row, int col",f=`b * ${p} + (row / 2) * ${d} + (col / 2)`;for(let g=2;g<a-1;g++)h=`int b${g}, `+h,p*=r[a-g-1],f=`b${g} * ${p} + `+f;return`
    vec4 ${s}(${h}) {
      int index = ${f};
      int texR = index / ${u};
      int texC = index - texR * ${u};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${u}, ${l});
      return ${o.texture2D}(${t}, uv);
    }
  `}function Up(n,e){const t=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=t[3],a=t[2]*r,c=t[1]*a,{newShape:i,keptDims:l}=Pe(t);if(i.length<t.length){const C=Je(n,i),$=["row","col","depth","depth2"];return`
      ${Qe(C,e)}
      float ${o}(int row, int col, int depth, int depth2) {
        return ${o}(${et($,l)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${o}(int row, int col, int depth, int depth2) {
        int index = round(dot(vec4(row, col, depth, depth2),
                          vec4(${c}, ${a}, ${r}, 1)));
        ${Ze(n)}
      }
    `;const u=n.shapeInfo.flatOffset,d=n.shapeInfo.texShape,p=d[0],h=d[1],f=`int stride2 = ${s}Shape[3];`,g=`int stride1 = ${s}Shape[2] * stride2;`,x=`int stride0 = ${s}Shape[1] * stride1;`;if(h===c&&u==null)return e?`
      float ${o}(int row, int col, int depth, int depth2) {
        ${f}
        ${g}
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(stride1, stride2, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
      float ${o}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(${a}, ${r}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${h}.0, ${p}.0);
        return sampleTexture(${s}, uv);
      }
    `;if(h===r&&u==null)return e?`
      float ${o}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${s}Shape[1] * ${s}Shape[2], ${s}Shape[2], 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
      float ${o}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${t[1]*t[2]}, ${t[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${h}.0, ${p}.0);
        return sampleTexture(${s}, uv);
      }
    `;const m=Ve(s);return e?`
    float ${o}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      ${f}
      ${g}
      ${x}
      int index = row * stride0 + col * stride1 +
          depth * stride2 + depth2;
      vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index + ${m});
      return sampleTexture(${s}, uv);
    }
  `:`
    float ${o}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${c} + col * ${a} +
          depth * ${r} + depth2;
      vec2 uv = uvFromFlat(${p}, ${h}, index + ${m});
      return sampleTexture(${s}, uv);
    }
  `}function Gp(n){const e=n.shapeInfo.logicalShape,t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1),o=e[4],r=e[3]*o,a=e[2]*r,c=e[1]*a,{newShape:i,keptDims:l}=Pe(e);if(i.length<e.length){const g=Je(n,i),x=["row","col","depth","depth2","depth3"];return`
      ${Qe(g)}
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        return ${s}(${et(x,l)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${c}, ${a}, ${r}, ${o})) +
          depth3;
        ${Ze(n)}
      }
    `;const u=n.shapeInfo.flatOffset,d=n.shapeInfo.texShape,p=d[0],h=d[1];if(h===c&&u==null)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
                         vec4(${a}, ${r}, ${o}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${h}.0, ${p}.0);
        return sampleTexture(${t}, uv);
      }
    `;if(h===o&&u==null)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${e[1]*e[2]*e[3]},
               ${e[2]*e[3]}, ${e[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${h}.0, ${p}.0);
        return sampleTexture(${t}, uv);
      }
    `;const f=Ve(t);return`
    float ${s}(int row, int col, int depth, int depth2, int depth3) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${c} + col * ${a} + depth * ${r} +
          depth2 * ${o} + depth3 + ${f};
      vec2 uv = uvFromFlat(${p}, ${h}, index);
      return sampleTexture(${t}, uv);
    }
  `}function zp(n){const e=n.shapeInfo.logicalShape,t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1),{newShape:o,keptDims:r}=Pe(e);if(o.length<e.length){const x=Je(n,o),m=["row","col","depth","depth2","depth3","depth4"];return`
      ${Qe(x)}
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${s}(${et(m,r)});
      }
    `}const a=e[5],c=e[4]*a,i=e[3]*c,l=e[2]*i,u=e[1]*l;if(n.shapeInfo.isUniform)return`
      float ${s}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        int index = round(dot(
          vec4(row, col, depth, depth2),
          vec4(${u}, ${l}, ${i}, ${c})) +
          dot(
            vec2(depth3, depth4),
            vec2(${a}, 1)));
        ${Ze(n)}
      }
    `;const d=n.shapeInfo.flatOffset,p=n.shapeInfo.texShape,h=p[0],f=p[1];if(f===u&&d==null)return`
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
          vec4(${l}, ${i}, ${c}, ${a})) +
               float(depth4);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${f}.0, ${h}.0);
        return sampleTexture(${t}, uv);
      }
    `;if(f===a&&d==null)return`
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(vec4(row, col, depth, depth2),
          vec4(${e[1]*e[2]*e[3]*e[4]},
               ${e[2]*e[3]*e[4]},
               ${e[3]*e[4]},
               ${e[4]})) + float(depth3);
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${f}.0, ${h}.0);
        return sampleTexture(${t}, uv);
      }
    `;const g=Ve(t);return`
    float ${s}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${u} + col * ${l} + depth * ${i} +
          depth2 * ${c} + depth3 * ${a} + depth4 + ${g};
      vec2 uv = uvFromFlat(${h}, ${f}, index);
      return sampleTexture(${t}, uv);
    }
  `}function Ze(n){const e=n.name,t=E(n.shapeInfo.logicalShape);return t<2?`return ${e};`:`
    for (int i = 0; i < ${t}; i++) {
      if (i == index) {
        return ${e}[i];
      }
    }
  `}function Hp(n,e){const t=n.name,s=t.charAt(0).toUpperCase()+t.slice(1),o="get"+s+"AtOutCoords",r=n.shapeInfo.logicalShape.length,a=e.logicalShape.length,c=ia(n.shapeInfo.logicalShape,e.logicalShape),i=_(a),l=a-r;let u;const d=["x","y","z","w","u","v"];r===0?u="":a<2&&c.length>=1?u="coords = 0;":u=c.map(C=>`coords.${d[C+l]} = 0;`).join(`
`);let p="";a<2&&r>0?p="coords":p=n.shapeInfo.logicalShape.map((C,$)=>`coords.${d[$+l]}`).join(", ");let h="return outputValue;";const g=E(n.shapeInfo.logicalShape)===1,m=E(e.logicalShape)===1;if(r===1&&!g&&!m)h=`
      return vec4(outputValue.xy, outputValue.xy);
    `;else if(g&&!m)a===1?h=`
        return vec4(outputValue.x, outputValue.x, 0., 0.);
      `:h=`
        return vec4(outputValue.x);
      `;else if(c.length){const C=r-2,$=r-1;c.indexOf(C)>-1&&c.indexOf($)>-1?h="return vec4(outputValue.x);":c.indexOf(C)>-1?h="return vec4(outputValue.x, outputValue.y, outputValue.x, outputValue.y);":c.indexOf($)>-1&&(h="return vec4(outputValue.xx, outputValue.zz);")}return`
    vec4 ${o}() {
      ${i} coords = getOutputCoords();
      ${u}
      vec4 outputValue = get${s}(${p});
      ${h}
    }
  `}function Xp(n,e){const t=n.name,s=t.charAt(0).toUpperCase()+t.slice(1),o="get"+s+"AtOutCoords",r=e.texShape,a=n.shapeInfo.texShape,c=n.shapeInfo.logicalShape.length,i=e.logicalShape.length;if(!n.shapeInfo.isUniform&&c===i&&n.shapeInfo.flatOffset==null&&q(a,r))return`
      float ${o}() {
        return sampleTexture(${t}, resultUV);
      }
    `;const l=_(i),u=ia(n.shapeInfo.logicalShape,e.logicalShape),d=i-c;let p;const h=["x","y","z","w","u","v"];c===0?p="":i<2&&u.length>=1?p="coords = 0;":p=u.map(g=>`coords.${h[g+d]} = 0;`).join(`
`);let f="";return i<2&&c>0?f="coords":f=n.shapeInfo.logicalShape.map((g,x)=>`coords.${h[x+d]}`).join(", "),`
    float ${o}() {
      ${l} coords = getOutputCoords();
      ${p}
      return get${s}(${f});
    }
  `}function _(n){if(n<=1)return"int";if(n===2)return"ivec2";if(n===3)return"ivec3";if(n===4)return"ivec4";if(n===5)return"ivec5";if(n===6)return"ivec6";throw Error(`GPU for rank ${n} is not yet supported`)}function as(n,e,t){const{newShape:s,keptDims:o}=Pe(e),r=e.length,a=n&&r===3&&e[0]===1,c=a?e.slice(1):s,i=!n&&r>1&&!q(e,t)&&s.length<r||a;return{useSqueezeShape:i,uniformShape:i?c:e,keptDims:o}}function Je(n,e){const t=JSON.parse(JSON.stringify(n));return t.shapeInfo.logicalShape=e,t}function et(n,e){return e.map(t=>n[t]).join(", ")}function Kp(n,e,t,s){const o=t.map((u,d)=>{const p={logicalShape:u.shape,texShape:u.isUniform?null:u.texData.texShape,isUniform:u.isUniform,isPacked:u.isUniform?!1:u.texData.isPacked,flatOffset:null};return u.texData!=null&&u.texData.slice!=null&&u.texData.slice.flatOffset>0&&(p.flatOffset=u.texData.slice.flatOffset),{name:e.variableNames[d],shapeInfo:p}}),r=o.map(u=>u.shapeInfo),a={logicalShape:s.shape,texShape:s.texData.texShape,isUniform:!1,isPacked:s.texData.isPacked,flatOffset:null},c=dp(o,a,e),i=Br(n.gl,c),l=n.createProgram(i);return w().get("ENGINE_COMPILE_ONLY")?{program:e,fragmentShader:i,source:c,webGLProgram:l,inShapeInfos:r,outShapeInfo:a,variablesLocations:null,customUniformLocations:null,infLoc:null,nanLoc:null,outShapeLocation:null,outShapeStridesLocation:null,outTexShapeLocation:null}:(n.buildVao(l),Object.assign({program:e,fragmentShader:i,source:c,webGLProgram:l,inShapeInfos:r,outShapeInfo:a},ua(n,e,l)))}function ua(n,e,t){const s=[],o=[];let r,a,c,i=null,l=null;l=n.getUniformLocation(t,"NAN",!1),w().getNumber("WEBGL_VERSION")===1&&(i=n.getUniformLocation(t,"INFINITY",!1));const u=!1;for(const d of e.variableNames){const p={name:d,uniform:n.getUniformLocation(t,d,u),offset:n.getUniformLocation(t,`offset${d}`,u)};e.enableShapeUniforms&&(p.shape=n.getUniformLocation(t,`${d}Shape`,u),p.texShape=n.getUniformLocation(t,`${d}TexShape`,u)),s.push(p)}if(e.enableShapeUniforms&&(r=n.getUniformLocation(t,"outShape",u),c=n.getUniformLocation(t,"outShapeStrides",u),a=n.getUniformLocation(t,"outTexShape",u)),e.customUniforms)for(const d of e.customUniforms)o.push(n.getUniformLocation(t,d.name,u));return{variablesLocations:s,customUniformLocations:o,infLoc:i,nanLoc:l,outShapeLocation:r,outShapeStridesLocation:c,outTexShapeLocation:a}}function Es(n,e){if(n.length!==e.length)throw Error(`Binary was compiled with ${n.length} inputs, but was executed with ${e.length} inputs`);n.forEach((t,s)=>{const o=t.logicalShape,r=e[s],a=r.shape;if(!q(o,a))throw Error(`Binary was compiled with different shapes than the current args. Shapes ${o} and ${a} must match`);if(t.isUniform&&r.isUniform)return;const c=t.texShape,i=r.isUniform?null:r.texData.texShape;if(!q(c,i))throw Error(`Binary was compiled with different texture shapes than the current args. Shape ${c} and ${i} must match`)})}function jp(n,e,t,s,o){e.program.enableShapeUniforms||(Es(e.inShapeInfos,t),Es([e.outShapeInfo],[s]));const r=s.texData.texture,a=s.texData.texShape;s.texData.isPacked?n.setOutputPackedMatrixTexture(r.texture,a[0],a[1]):n.setOutputMatrixTexture(r.texture,a[0],a[1]),n.setProgram(e.webGLProgram),n.bindVertexArray(e.webGLProgram.vao),w().getNumber("WEBGL_VERSION")===1&&e.infLoc!==null&&n.gl.uniform1f(e.infLoc,1/0),e.nanLoc!==null&&n.gl.uniform1f(e.nanLoc,NaN);for(let i=0;i<t.length;++i){const l=t[i],{uniform:u,offset:d,shape:p,texShape:h}=e.variablesLocations[i];if(p){const{uniformShape:f}=as(e.program.packedInputs,l.shape,l.texData.texShape);switch(f.length){case 1:n.gl.uniform1iv(p,new Int32Array(f));break;case 2:n.gl.uniform2iv(p,new Int32Array(f));break;case 3:n.gl.uniform3iv(p,new Int32Array(f));break;case 4:n.gl.uniform4iv(p,new Int32Array(f));break}}if(h&&n.gl.uniform2i(h,l.texData.texShape[0],l.texData.texShape[1]),u!=null){if(l.isUniform){if(E(l.shape)<2)n.gl.uniform1f(u,l.uniformValues[0]);else{let f=l.uniformValues;f instanceof Float32Array||(f=new Float32Array(f)),n.gl.uniform1fv(u,f)}continue}l.texData.slice!=null&&d!=null&&n.gl.uniform1i(d,l.texData.slice.flatOffset),n.setInputMatrixTexture(l.texData.texture.texture,u,i)}}const c=e.outShapeLocation;if(c)switch(s.shape.length){case 1:n.gl.uniform1iv(c,new Int32Array(s.shape));break;case 2:n.gl.uniform2iv(c,new Int32Array(s.shape));break;case 3:n.gl.uniform3iv(c,new Int32Array(s.shape));break;case 4:n.gl.uniform4iv(c,new Int32Array(s.shape));break}if(e.outShapeStridesLocation){const i=M(s.shape);switch(s.shape.length){case 2:n.gl.uniform1iv(e.outShapeStridesLocation,new Int32Array(i));break;case 3:n.gl.uniform2iv(e.outShapeStridesLocation,new Int32Array(i));break;case 4:n.gl.uniform3iv(e.outShapeStridesLocation,new Int32Array(i));break}}if(e.outTexShapeLocation&&n.gl.uniform2i(e.outTexShapeLocation,s.texData.texShape[0],s.texData.texShape[1]),e.program.customUniforms&&o)for(let i=0;i<e.program.customUniforms.length;++i){const l=e.program.customUniforms[i],u=e.customUniformLocations[i],d=o[i];if(l.type==="float")n.gl.uniform1fv(u,d);else if(l.type==="vec2")n.gl.uniform2fv(u,d);else if(l.type==="vec3")n.gl.uniform3fv(u,d);else if(l.type==="vec4")n.gl.uniform4fv(u,d);else if(l.type==="int")n.gl.uniform1iv(u,d);else if(l.type==="ivec2")n.gl.uniform2iv(u,d);else if(l.type==="ivec3")n.gl.uniform3iv(u,d);else if(l.type==="ivec4")n.gl.uniform4iv(u,d);else throw Error(`uniform type ${l.type} is not supported yet.`)}n.executeProgram()}function qp(n,e,t){let s="";e.concat(t).forEach(a=>{const c=a.texData!=null&&a.texData.slice!=null&&a.texData.slice.flatOffset>0;if(n.enableShapeUniforms&&!a.isUniform){const i=a.texData.texShape,{useSqueezeShape:l,uniformShape:u,keptDims:d}=as(n.packedInputs,a.shape,i);let p="",h="",f="";if(u.length===1&&n.packedInputs){const v=[Math.ceil(i[0]/2),Math.ceil(i[1]/2)];p=`${v[0]>1}_${v[1]>1}`}else if(u.length===2&&!n.packedInputs)h=`${u[0]>1}_${u[1]>1}`;else if(u.length>2&&!n.packedInputs){const v=M(u);f=`${v[0]===i[1]}_${v[v.length-1]===i[1]}`}const g=a.shape.length,x=u.length===2&&q(a.shape,i),m=E(a.shape)===1,C=ze(a.shape,t.shape),$=!n.packedInputs&&g===t.shape.length&&q(i,t.texData.texShape),b=n.packedInputs||u.length>2?"":`${i[0]>1}_${i[1]>1}`;s+=`${g}_${$}_${l?d:""}_${u.length}_${m}_${C}_${x}_${p}_${h}_${f}_${b}_${c}`}else{const i=a.isUniform?"uniform":a.texData.texShape;s+=`${a.shape}_${i}_${c}`}});const o=n.userCode;let r=n.constructor.name;return r+="_"+s+"_"+o+`${w().getNumber("WEBGL_VERSION")}`,r}function W(n){return w().getBool("WEBGL_USE_SHAPES_UNIFORMS")&&n<=4}class Yp{constructor(e){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outPackingScheme=ft.DENSE,this.customUniforms=[{name:"texShape",type:"ivec2"}];const t=K();this.outputShape=e,this.enableShapeUniforms=W(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?Yt(["r","c","d"],e):Le(["r","c","d"],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getA(rc.x, rc.y, rc.z);
        }

        ${t.output} = result;
      }
    `}}class Qp{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outPackingScheme=ft.DENSE,this.customUniforms=[{name:"texShape",type:"ivec2"}];const t=K();this.outputShape=e,this.enableShapeUniforms=W(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?Yt(["r","c","d"],e):Le(["r","c","d"],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getChannel(getA(rc.x, rc.y, rc.z), vec2(rc.y, rc.z));
        }

        ${t.output} = result;
      }
    `}}class Zp{constructor(e){this.variableNames=["A"],this.outTexUsage=te.DOWNLOAD;const t=K();this.outputShape=e,this.userCode=`
      ${aa}

      void main() {
        float x = getAAtOutCoords();
        ${t.output} = encode_float(x);
      }
    `}}class Jp{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outTexUsage=te.DOWNLOAD;const t=K();this.outputShape=e,this.userCode=`
      ${aa}

      void main() {
        ivec3 coords = getOutputCoords();
        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));
        ${t.output} = encode_float(x);
      }
    `}}const eh={R:0,G:1,B:2,A:3};class Ns{constructor(e,t=!1,s="RGBA"){this.variableNames=["A"],this.customUniforms=[{name:"texShape",type:"ivec2"}];const o=K();this.outputShape=e,this.enableShapeUniforms=W(this.outputShape.length);let r="result";t&&(r="floor(result * 255. + 0.5)");let a="";for(let c=0;c<s.length;c++){const i=s[c];a+=`
          if(offset == ${c}) {
            result = values[${eh[i]}];
          }`}this.userCode=`
      ${this.enableShapeUniforms?rs():os(e)}

      void main() {
        ivec3 coords = getOutputCoords();
        int flatIndex = getFlatIndex(coords);
        float result = 0.;
        int offset = imod(flatIndex, ${s.length});

        flatIndex = idiv(flatIndex, ${s.length}, 1.);

        int r = flatIndex / texShape[1];
        if (r < texShape[0]) {
          int c = imod(flatIndex, texShape[1]);
          vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
          vec4 values = ${o.texture2D}(A, uv);
          ${a}
        }
        ${o.output} = vec4(${r}, 0., 0., 0.);
      }
    `}}class th{constructor(e,t=!1){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.customUniforms=[{name:"texShape",type:"ivec2"}];const s=K();this.outputShape=e,this.enableShapeUniforms=W(this.outputShape.length);let o="",r="result";t&&(r="floor(result * 255. + 0.5)");for(let a=0;a<=1;a++)for(let c=0;c<=1;c++){const i=a*2+c;o+=`
          localCoords = coords;
          if(localCoords[2] + ${c} < ${this.enableShapeUniforms?"outShape[2]":`${e[2]}`}) {
          localCoords[2] += ${c};
          if (localCoords[1] + ${a} < ${this.enableShapeUniforms?"outShape[1]":`${e[1]}`}) {
            localCoords[1] += ${a};

            flatIndex = getFlatIndex(localCoords);
            offset = imod(flatIndex, 4);

            flatIndex = idiv(flatIndex, 4, 1.);

            int r = flatIndex / texShape[1];
            int c = imod(flatIndex, texShape[1]);
            vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
            values = ${s.texture2D}(A, uv);

            if (offset == 0) {
              result[${i}] = values[0];
            } else if (offset == 1) {
              result[${i}] = values[1];
            } else if (offset == 2) {
              result[${i}] = values[2];
            } else {
              result[${i}] = values[3];
            }
          }
        }
        `}this.userCode=`
        ${this.enableShapeUniforms?rs():os(e)}

        void main() {
          ivec3 coords = getOutputCoords();

          vec4 result = vec4(0.);
          int flatIndex, r, c, offset;
          ivec3 localCoords;
          vec2 uv;
          vec4 values;

          ${o}

          ${s.output} = ${r};
        }
    `}}function da(n){const e=K(),t=`${e.version}
    precision highp float;
    ${e.attribute} vec3 clipSpacePos;
    ${e.attribute} vec2 uv;
    ${e.varyingVs} vec2 resultUV;

    void main() {
      gl_Position = vec4(clipSpacePos, 1);
      resultUV = uv;
    }`;return Vr(n,t)}function pa(n){const e=new Float32Array([-1,1,0,0,1,-1,-1,0,0,0,1,1,0,1,1,1,-1,0,1,0]);return Ur(n,e)}function ha(n){const e=new Uint16Array([0,1,2,2,1,3]);return Gr(n,e)}function It(n,e,t,s,o,r){Hr(e,t);const a=zr(n),c=n.TEXTURE_2D;return y(n,()=>n.bindTexture(c,a)),y(n,()=>n.texParameteri(c,n.TEXTURE_WRAP_S,n.CLAMP_TO_EDGE)),y(n,()=>n.texParameteri(c,n.TEXTURE_WRAP_T,n.CLAMP_TO_EDGE)),y(n,()=>n.texParameteri(c,n.TEXTURE_MIN_FILTER,n.NEAREST)),y(n,()=>n.texParameteri(c,n.TEXTURE_MAG_FILTER,n.NEAREST)),w().getNumber("WEBGL_VERSION")===1?y(n,()=>n.texImage2D(c,0,s,e,t,0,o,r,null)):y(n,()=>n.texStorage2D(c,1,s,e,t)),y(n,()=>n.bindTexture(n.TEXTURE_2D,null)),{texture:a,texShape:[t,e]}}function is(n){return n.internalFormatFloat}function fa(n,e,t,s){const[o,r]=wt(e,t);return It(n,o,r,is(s),s.textureFormatFloat,n.FLOAT)}function cs(n){return n.internalFormatHalfFloat}function ma(n,e,t,s){const[o,r]=wt(e,t);return It(n,o,r,cs(s),s.textureFormatFloat,s.textureTypeHalfFloat)}function ls(n){return n.downloadTextureFormat}function xa(n,e,t,s){const[o,r]=wt(e,t);return It(n,o,r,ls(s),n.RGBA,n.UNSIGNED_BYTE)}function us(n){return n.internalFormatPackedFloat}function ga(n,e,t,s){const[o,r]=qe(e,t);return It(n,o,r,us(s),n.RGBA,n.FLOAT)}function ds(n){return n.internalFormatPackedHalfFloat}function Ca(n,e,t,s){const[o,r]=qe(e,t);return It(n,o,r,ds(s),n.RGBA,s.textureTypeHalfFloat)}function $a(n,e,t){return y(n,()=>n.bindBuffer(n.ARRAY_BUFFER,t)),pn(n,e,"clipSpacePos",t,3,20,0)&&pn(n,e,"uv",t,2,20,12)}function ba(n,e,t,s,o,r){y(n,()=>n.bindTexture(n.TEXTURE_2D,e));let a,c,i;o instanceof Uint8Array?(a=new Uint8Array(t*s*4),c=n.UNSIGNED_BYTE,i=n.RGBA):(a=new Float32Array(t*s*4),c=n.FLOAT,i=r.internalFormatPackedFloat),a.set(o),w().getNumber("WEBGL_VERSION")===2?y(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,t,s,n.RGBA,c,a)):y(n,()=>n.texImage2D(n.TEXTURE_2D,0,i,t,s,0,n.RGBA,c,a)),y(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function va(n,e,t){y(n,()=>n.bindTexture(n.TEXTURE_2D,e)),t.data instanceof Uint8Array?w().getNumber("WEBGL_VERSION")===2?y(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,t.width,t.height,n.RGBA,n.UNSIGNED_BYTE,t.data)):y(n,()=>n.texImage2D(n.TEXTURE_2D,0,n.RGBA,t.width,t.height,0,n.RGBA,n.UNSIGNED_BYTE,t.data)):w().getNumber("WEBGL_VERSION")===2?y(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,n.RGBA,n.UNSIGNED_BYTE,t)):y(n,()=>n.texImage2D(n.TEXTURE_2D,0,n.RGBA,n.RGBA,n.UNSIGNED_BYTE,t)),y(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function wa(n,e,t,s){const o=n.createBuffer();y(n,()=>n.bindBuffer(n.PIXEL_PACK_BUFFER,o));const c=4*4*e*t;return y(n,()=>n.bufferData(n.PIXEL_PACK_BUFFER,c,n.STREAM_READ)),y(n,()=>n.readPixels(0,0,t,e,n.RGBA,n.FLOAT,0)),y(n,()=>n.bindBuffer(n.PIXEL_PACK_BUFFER,null)),o}function Ia(n,e,t){const s=n,o=new Float32Array(t);return s.bindBuffer(s.PIXEL_PACK_BUFFER,e),s.getBufferSubData(s.PIXEL_PACK_BUFFER,0,o),s.bindBuffer(s.PIXEL_PACK_BUFFER,null),o}function Ra(n,e,t,s){const[o,r]=wt(e,t),a=4,c=new Uint8Array(Yd(e*t,a));return y(n,()=>n.readPixels(0,0,o,r,s.downloadTextureFormat,n.UNSIGNED_BYTE,c)),new Float32Array(c.buffer)}function ya(n,e,t,s,o,r,a,c){const i=n,l=new Float32Array(Qd(r,a));return i.bindBuffer(i.PIXEL_PACK_BUFFER,e),i.getBufferSubData(i.PIXEL_PACK_BUFFER,0,l),i.bindBuffer(i.PIXEL_PACK_BUFFER,null),l}function Sa(n,e,t){const s=new Float32Array(e*t*4);return y(n,()=>n.readPixels(0,0,t,e,n.RGBA,n.FLOAT,s)),s}const nh=Object.freeze(Object.defineProperty({__proto__:null,bindVertexProgramAttributeStreams:$a,createBufferFromOutputTexture:wa,createFloat16MatrixTexture:ma,createFloat16PackedMatrixTexture:Ca,createFloat32MatrixTexture:fa,createIndexBuffer:ha,createPackedMatrixTexture:ga,createUnsignedBytesMatrixTexture:xa,createVertexBuffer:pa,createVertexShader:da,downloadByteEncodedFloatMatrixFromOutputTexture:Ra,downloadFloat32MatrixFromBuffer:Ia,downloadMatrixFromPackedOutputTexture:Sa,downloadPackedMatrixFromBuffer:ya,getInternalFormatForFloat16MatrixTexture:cs,getInternalFormatForFloat16PackedMatrixTexture:ds,getInternalFormatForFloat32MatrixTexture:is,getInternalFormatForPackedMatrixTexture:us,getInternalFormatForUnsignedBytesMatrixTexture:ls,uploadDenseMatrixToTexture:ba,uploadPixelDataToTexture:va},Symbol.toStringTag,{value:"Module"}));class Lt{constructor(e){this.outputTexture=null,this.program=null,this.disposed=!1,this.itemsToPoll=[];const t=w().getNumber("WEBGL_VERSION");if(e!=null?(this.gl=e,Pr(t,e)):this.gl=ue(t),e=this.gl,w().getNumber("WEBGL_VERSION")===2){const r=e;this.createVertexArray=()=>y(r,()=>r.createVertexArray()),this.bindVertexArray=a=>y(r,()=>r.bindVertexArray(a)),this.deleteVertexArray=a=>y(r,()=>r.deleteVertexArray(a)),this.getVertexArray=()=>y(r,()=>r.getParameter(r.VERTEX_ARRAY_BINDING))}else if(e!=null){const r=e.getExtension("OES_vertex_array_object");if(r==null)throw new Error("All WebGL1 implementations are expected to offer OES_vertex_array_object.");this.createVertexArray=()=>y(e,()=>r.createVertexArrayOES()),this.bindVertexArray=a=>y(e,()=>r.bindVertexArrayOES(a)),this.deleteVertexArray=a=>y(e,()=>r.deleteVertexArrayOES(a)),this.getVertexArray=()=>y(e,()=>e.getParameter(r.VERTEX_ARRAY_BINDING_OES))}let s="WEBGL_color_buffer_float";const o="EXT_color_buffer_half_float";if(this.parallelCompilationExtension=this.gl.getExtension("KHR_parallel_shader_compile"),w().getNumber("WEBGL_VERSION")===1){const r="OES_texture_float",a="OES_texture_half_float";if(this.textureFloatExtension=at(this.gl,r),ne(this.gl,a))this.textureHalfFloatExtension=at(this.gl,a);else if(w().get("WEBGL_FORCE_F16_TEXTURES"))throw new Error("GL context does not support half float textures, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.");if(this.colorBufferFloatExtension=this.gl.getExtension(s),ne(this.gl,o))this.colorBufferHalfFloatExtension=at(this.gl,o);else if(w().get("WEBGL_FORCE_F16_TEXTURES"))throw new Error("GL context does not support color renderable half floats, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.")}else if(s="EXT_color_buffer_float",ne(this.gl,s))this.colorBufferFloatExtension=this.gl.getExtension(s);else if(ne(this.gl,o))this.colorBufferHalfFloatExtension=this.gl.getExtension(o);else throw new Error("GL context does not support color renderable floats");this.vertexBuffer=pa(this.gl),this.indexBuffer=ha(this.gl),this.framebuffer=Xr(this.gl),this.textureConfig=ns(this.gl,this.textureHalfFloatExtension)}get debug(){return w().getBool("DEBUG")}dispose(){if(this.disposed)return;this.program!=null&&console.warn("Disposing a GPGPUContext that still has a bound WebGLProgram. This is probably a resource leak, delete the program with GPGPUContext.deleteProgram before disposing."),this.outputTexture!=null&&console.warn("Disposing a GPGPUContext that still has a bound output matrix texture.  This is probably a resource leak, delete the output matrix texture with GPGPUContext.deleteMatrixTexture before disposing.");const e=this.gl;y(e,()=>e.finish()),y(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,null)),y(e,()=>e.deleteFramebuffer(this.framebuffer)),y(e,()=>e.bindBuffer(e.ARRAY_BUFFER,null)),y(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,null)),y(e,()=>e.deleteBuffer(this.indexBuffer)),this.disposed=!0}createFloat32MatrixTexture(e,t){return this.throwIfDisposed(),fa(this.gl,e,t,this.textureConfig)}createFloat16MatrixTexture(e,t){return this.throwIfDisposed(),ma(this.gl,e,t,this.textureConfig)}createUnsignedBytesMatrixTexture(e,t){return this.throwIfDisposed(),xa(this.gl,e,t,this.textureConfig)}uploadPixelDataToTexture(e,t){this.throwIfDisposed(),va(this.gl,e,t)}uploadDenseMatrixToTexture(e,t,s,o){this.throwIfDisposed(),ba(this.gl,e,t,s,o,this.textureConfig)}createFloat16PackedMatrixTexture(e,t){return this.throwIfDisposed(),Ca(this.gl,e,t,this.textureConfig)}createPackedMatrixTexture(e,t){return this.throwIfDisposed(),ga(this.gl,e,t,this.textureConfig)}deleteMatrixTexture(e){this.throwIfDisposed(),this.outputTexture===e&&(hn(this.gl,this.framebuffer),this.outputTexture=null),y(this.gl,()=>this.gl.deleteTexture(e))}downloadByteEncodedFloatMatrixFromOutputTexture(e,t,s){return this.downloadMatrixDriver(e,()=>Ra(this.gl,t,s,this.textureConfig))}downloadPackedMatrixFromBuffer(e,t,s,o,r,a){return ya(this.gl,e,t,s,o,r,a,this.textureConfig)}downloadFloat32MatrixFromBuffer(e,t){return Ia(this.gl,e,t)}createBufferFromTexture(e,t,s){this.bindTextureToFrameBuffer(e);const o=wa(this.gl,t,s,this.textureConfig);return this.unbindTextureToFrameBuffer(),o}createAndWaitForFence(){const e=this.createFence(this.gl);return this.pollFence(e)}createFence(e){let t,s;if(w().getBool("WEBGL_FENCE_API_ENABLED")){const o=e,r=o.fenceSync(o.SYNC_GPU_COMMANDS_COMPLETE,0);e.flush(),s=()=>{const a=o.clientWaitSync(r,0,0);return a===o.ALREADY_SIGNALED||a===o.CONDITION_SATISFIED},t=r}else w().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0?(t=this.beginQuery(),this.endQuery(),s=()=>this.isQueryAvailable(t,w().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))):s=()=>!0;return{query:t,isFencePassed:s}}downloadMatrixFromPackedTexture(e,t,s){return this.downloadMatrixDriver(e,()=>Sa(this.gl,t,s))}createProgram(e){this.throwIfDisposed();const t=this.gl;this.vertexShader==null&&(this.vertexShader=da(t));const s=Mr(t);y(t,()=>t.attachShader(s,this.vertexShader)),y(t,()=>t.attachShader(s,e)),Wr(t,s);const o=Object.assign(s,{vao:this.createVertexArray()});return this.debug&&Dt(t,o),o}buildVao(e){this.setProgram(e),this.bindVertexArray(e.vao);const t=this.gl;y(t,()=>t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,this.indexBuffer)),$a(t,e,this.vertexBuffer)}deleteProgram(e){this.throwIfDisposed(),e===this.program&&(this.program=null),e!=null&&(y(this.gl,()=>this.gl.deleteProgram(e)),this.deleteVertexArray(e.vao))}setProgram(e){this.throwIfDisposed(),this.program=e,this.program!=null&&this.debug&&Dt(this.gl,this.program),y(this.gl,()=>this.gl.useProgram(e))}getUniformLocation(e,t,s=!0){return this.throwIfDisposed(),s?jr(this.gl,e,t):qr(this.gl,e,t)}getAttributeLocation(e,t){return this.throwIfDisposed(),y(this.gl,()=>this.gl.getAttribLocation(e,t))}getUniformLocationNoThrow(e,t){return this.throwIfDisposed(),this.gl.getUniformLocation(e,t)}setInputMatrixTexture(e,t,s){this.throwIfDisposed(),this.throwIfNoProgram(),Yr(this.gl,e,t,s)}setOutputMatrixTexture(e,t,s){this.setOutputMatrixTextureDriver(e,s,t)}setOutputPackedMatrixTexture(e,t,s){this.throwIfDisposed();const[o,r]=qe(t,s);this.setOutputMatrixTextureDriver(e,o,r)}setOutputMatrixWriteRegion(e,t,s,o){this.setOutputMatrixWriteRegionDriver(s,e,o,t)}setOutputPackedMatrixWriteRegion(e,t,s,o){throw new Error("setOutputPackedMatrixWriteRegion not implemented.")}debugValidate(){this.program!=null&&Dt(this.gl,this.program),it(this.gl)}executeProgram(){this.throwIfDisposed(),this.throwIfNoProgram();const e=this.gl;if(this.debug){const t=this.getVertexArray();console.assert(t===this.program.vao,"VAO changed between setProgram and executeProgram!"),this.debugValidate()}y(e,()=>e.drawElements(e.TRIANGLES,6,e.UNSIGNED_SHORT,0))}blockUntilAllProgramsCompleted(){this.throwIfDisposed(),y(this.gl,()=>this.gl.finish())}getQueryTimerExtension(){return this.disjointQueryTimerExtension==null&&(this.disjointQueryTimerExtension=at(this.gl,w().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2?"EXT_disjoint_timer_query_webgl2":"EXT_disjoint_timer_query")),this.disjointQueryTimerExtension}getQueryTimerExtensionWebGL2(){return this.getQueryTimerExtension()}getQueryTimerExtensionWebGL1(){return this.getQueryTimerExtension()}beginQuery(){if(w().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2){const s=this.gl,o=this.getQueryTimerExtensionWebGL2(),r=s.createQuery();return s.beginQuery(o.TIME_ELAPSED_EXT,r),r}const e=this.getQueryTimerExtensionWebGL1(),t=e.createQueryEXT();return e.beginQueryEXT(e.TIME_ELAPSED_EXT,t),t}endQuery(){if(w().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2){const t=this.gl,s=this.getQueryTimerExtensionWebGL2();t.endQuery(s.TIME_ELAPSED_EXT);return}const e=this.getQueryTimerExtensionWebGL1();e.endQueryEXT(e.TIME_ELAPSED_EXT)}async waitForQueryAndGetTime(e){return await $s(()=>this.disposed||this.isQueryAvailable(e,w().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))),this.getQueryTime(e,w().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))}getQueryTime(e,t){if(t===0)return null;if(t===2){const s=this.gl;return s.getQueryParameter(e,s.QUERY_RESULT)/1e6}else{const s=this.getQueryTimerExtensionWebGL1();return s.getQueryObjectEXT(e,s.QUERY_RESULT_EXT)/1e6}}isQueryAvailable(e,t){if(t===0)return!0;if(t===2){const s=this.gl,o=this.getQueryTimerExtensionWebGL2(),r=s.getQueryParameter(e,s.QUERY_RESULT_AVAILABLE);return this.disjoint==null&&(this.disjoint=this.gl.getParameter(o.GPU_DISJOINT_EXT)),r&&!this.disjoint}else{const s=this.getQueryTimerExtensionWebGL1(),o=s.getQueryObjectEXT(e,s.QUERY_RESULT_AVAILABLE_EXT);return this.disjoint==null&&(this.disjoint=this.gl.getParameter(s.GPU_DISJOINT_EXT)),o&&!this.disjoint}}pollFence(e){return new Promise(t=>{this.addItemToPoll(()=>e.isFencePassed(),()=>t())})}pollItems(){const e=sh(this.itemsToPoll.map(t=>t.isDoneFn));for(let t=0;t<=e;++t){const{resolveFn:s}=this.itemsToPoll[t];s()}this.itemsToPoll=this.itemsToPoll.slice(e+1)}addItemToPoll(e,t){if(this.itemsToPoll.push({isDoneFn:e,resolveFn:t}),this.itemsToPoll.length>1)return;let s;"setTimeoutCustom"in w().platform&&(s=w().platform.setTimeoutCustom.bind(w().platform)),$s(()=>(this.pollItems(),this.itemsToPoll.length===0),()=>0,null,s)}bindTextureToFrameBuffer(e){this.throwIfDisposed(),Ft(this.gl,e,this.framebuffer),this.debug&&it(this.gl)}unbindTextureToFrameBuffer(){this.outputTexture!=null?(Ft(this.gl,this.outputTexture,this.framebuffer),this.debug&&it(this.gl)):hn(this.gl,this.framebuffer)}downloadMatrixDriver(e,t){this.bindTextureToFrameBuffer(e);const s=t();return this.unbindTextureToFrameBuffer(),s}setOutputMatrixTextureDriver(e,t,s){this.throwIfDisposed();const o=this.gl;Ft(o,e,this.framebuffer),this.debug&&it(o),this.outputTexture=e,y(o,()=>o.viewport(0,0,t,s)),y(o,()=>o.scissor(0,0,t,s))}setOutputMatrixWriteRegionDriver(e,t,s,o){this.throwIfDisposed(),y(this.gl,()=>this.gl.scissor(e,t,s,o))}throwIfDisposed(){if(this.disposed)throw new Error("Attempted to use disposed GPGPUContext.")}throwIfNoProgram(){if(this.program==null)throw new Error("No GPU program is currently set.")}}function sh(n){let e=0;for(;e<n.length&&n[e]();++e);return e-1}const{addImpl:oh,bincountImpl:Ta,bincountReduceImpl:rh,bitwiseAndImpl:ah,castImpl:ih,ceilImpl:ch,concatImpl:lh,equalImpl:uh,expImpl:dh,expm1Impl:ph,floorImpl:hh,gatherNdImpl:fh,gatherV2Impl:mh,greaterImpl:xh,greaterEqualImpl:gh,lessImpl:Ch,lessEqualImpl:$h,linSpaceImpl:bh,logImpl:vh,maxImpl:wh,maximumImpl:Ih,minimumImpl:Rh,multiplyImpl:yh,negImpl:Sh,notEqualImpl:Th,prodImpl:Eh,raggedGatherImpl:Nh,raggedRangeImpl:kh,raggedTensorToTensorImpl:Ah,rangeImpl:Oh,rsqrtImpl:Dh,scatterImpl:Fh,sigmoidImpl:Ph,simpleAbsImpl:Ea,sliceImpl:_h,sparseFillEmptyRowsImpl:Lh,sparseReshapeImpl:Vh,sparseSegmentReductionImpl:Na,sqrtImpl:Bh,staticRegexReplaceImpl:Mh,stridedSliceImpl:Wh,stringNGramsImpl:Uh,stringSplitImpl:Gh,stringToHashBucketFastImpl:zh,subImpl:Hh,tileImpl:Xh,topKImpl:Kh,transposeImpl:ps,uniqueImpl:jh}=Kd;function ka(n,e){return["x","y","z","w","u","v"].slice(0,e).map(t=>`${n}.${t}`)}function G(n,e){return e===1?[n]:ka(n,e)}function qh(n,e){if(n===1)return"rc";let t="";for(let s=0;s<n;s++)t+=e[s],s<n-1&&(t+=",");return t}class Yh{constructor(e){if(this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.enableShapeUniforms=W(this.outputShape.length),this.rank===0)this.userCode=`
        void main() {
          setOutput(vec4(getA(), 0., 0., 0.));
        }
      `;else{const t=G("rc",this.rank),s=_(this.rank),o=this.getOutOfBoundsCondition(t),r=this.getSetup(t),a=this.getOutput(t);this.userCode=`
        void main() {
          ${s} rc = getOutputCoords();

          if(${o}) {
            setOutput(vec4(0));
          } else {
            ${r}

            setOutput(vec4(${a}));
          }
        }
      `}}getSourceCoordsArr(e){const t=[];for(let s=0;s<=1;s++)for(let o=0;o<=1;o++){let r=`${s===0?"r":"rp1"}, ${o===0?"c":"cp1"}`;for(let a=2;a<this.rank;a++)r=`${e[e.length-1-a]},`+r;t.push(r)}return t}getOutOfBoundsCondition(e){if(this.rank===1)return`rc > ${this.enableShapeUniforms?"outShape":this.outputShape[0]}`;let t="";for(let s=this.rank-2;s<this.rank;s++)t+=`${e[s]} >= ${this.enableShapeUniforms?`outShape[${s}]`:this.outputShape[s]}`,s<this.rank-1&&(t+="||");return t}getSetup(e){if(this.rank===1)return"";const t=e.slice(-2),s=this.enableShapeUniforms?`outShape[${this.rank} - 1]`:this.outputShape[this.rank-1],o=this.enableShapeUniforms?`outShape[${this.rank} - 2]`:this.outputShape[this.rank-2];return`
      int r = ${t[0]};
      int c = ${t[1]};
      int rp1 = r + 1;
      int cp1 = c + 1;

      bool cEdge = cp1 >= ${s};
      bool rEdge = rp1 >= ${o};
    `}getOutput(e){const t=this.getSourceCoordsArr(e);return this.rank===1?`getA(rc), (rc + 1 >= ${this.enableShapeUniforms?"outShape":this.outputShape[0]} ? 0. : getA(rc + 1)), 0, 0`:`getA(${t[0]}),
            cEdge ? 0. : getA(${t[1]}),
            rEdge ? 0. : getA(${t[2]}),
            rEdge || cEdge ? 0. : getA(${t[3]})`}}class Aa{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"inputShape",type:"ivec3"}],this.outputShape=e,this.enableShapeUniforms=W(this.outputShape.length);let s="";for(let o=0;o<4;o++){let r="thisRC = rc;";o%2===1&&(r+="thisRC.z += 1;"),o>1&&(r+="thisRC.y += 1;"),s+=`
        ${r}
        ${o>0?"if(thisRC.y < rows && thisRC.z < cols){":""}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${o}] =
            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
        ${o>0?"}":""}
      `}this.userCode=`
      ${Qh(t,this.enableShapeUniforms)}
      ${this.enableShapeUniforms?rs():os(e)}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.);

        ivec3 thisRC;
        int rows = ${this.enableShapeUniforms?"outShape[1]":e[1]};
        int cols = ${this.enableShapeUniforms?"outShape[2]":e[2]};

        ${s}

        setOutput(result);
      }
    `}}function Qh(n,e){return`
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${e?up(["r","c","d"],"inputShape"):Le(["r","c","d"],n)}
      return ivec3(r, c, d);
    }
  `}class Zh{constructor(e){this.gpgpu=e,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0,this.freeTextures={},this.usedTextures={},this.logEnabled=!1}acquireTexture(e,t,s){const o=As(t,s),r=Os(e,o,s);r in this.freeTextures||(this.freeTextures[r]=[]),r in this.usedTextures||(this.usedTextures[r]=[]);const a=ks(e,o,this.gpgpu.gl,this.gpgpu.textureConfig,s);if(this.freeTextures[r].length>0){this.numFreeTextures--,this.numUsedTextures++,this._numBytesFree-=a,this.log();const i=this.freeTextures[r].pop();return this.usedTextures[r].push(i),i}let c;return o===L.PACKED_2X2_FLOAT32?c=this.gpgpu.createPackedMatrixTexture(e[0],e[1]):o===L.PACKED_2X2_FLOAT16?c=this.gpgpu.createFloat16PackedMatrixTexture(e[0],e[1]):o===L.UNPACKED_FLOAT32?c=this.gpgpu.createFloat32MatrixTexture(e[0],e[1]):o===L.UNPACKED_FLOAT16?c=this.gpgpu.createFloat16MatrixTexture(e[0],e[1]):o===L.PACKED_4X1_UNSIGNED_BYTE&&(c=this.gpgpu.createUnsignedBytesMatrixTexture(e[0],e[1])),this.usedTextures[r].push(c),this.numUsedTextures++,this._numBytesAllocated+=a,this.log(),c}releaseTexture(e,t,s,o){if(this.freeTextures==null)return;const r=As(s,o),a=Os(t,r,o);a in this.freeTextures||(this.freeTextures[a]=[]);const c=ks(t,r,this.gpgpu.gl,this.gpgpu.textureConfig,o),i=w().getNumber("WEBGL_DELETE_TEXTURE_THRESHOLD");i!==-1&&this._numBytesAllocated>i?(this.gpgpu.deleteMatrixTexture(e.texture),this._numBytesAllocated-=c):(this.freeTextures[a].push(e),this.numFreeTextures++,this._numBytesFree+=c),this.numUsedTextures--;const l=this.usedTextures[a],u=l&&l.indexOf(e);if(u==null||u<0)throw new Error("Cannot release a texture that was never provided by this texture manager");l[u]=l[l.length-1],l.pop(),this.log()}log(){if(!this.logEnabled)return;const e=this.numFreeTextures+this.numUsedTextures;console.log("Free/Used",`${this.numFreeTextures} / ${this.numUsedTextures}`,`(${e})`);const t=this._numBytesFree/this._numBytesAllocated;console.log(`Bytes allocated: ${this._numBytesAllocated}`),console.log(`Bytes unused: ${this._numBytesFree} (${Math.round(100*t)}%)`)}get numBytesAllocated(){return this._numBytesAllocated}get numBytesFree(){return this._numBytesFree}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){if(this.freeTextures!=null){for(const e in this.freeTextures)this.freeTextures[e].forEach(t=>{this.gpgpu.deleteMatrixTexture(t.texture)});for(const e in this.usedTextures)this.usedTextures[e].forEach(t=>{this.gpgpu.deleteMatrixTexture(t.texture)});this.freeTextures=null,this.usedTextures=null,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0}}}function Jh(n,e){const t=n;if(e===t.R32F)return 4;if(e===t.R16F)return 2;if(e===t.RGBA32F)return 16;if(e===n.RGBA)return 16;if(e===t.RGBA16F)return 8;if(e===t.RGBA8)return 4;throw new Error(`Unknown internal format ${e}`)}function ks(n,e,t,s,o){const r=ef(e,s);let a;if(o){const[i,l]=qe(n[0],n[1]);a=i*l}else{const[i,l]=wt(n[0],n[1]);a=i*l}const c=Jh(t,r);return a*c}function ef(n,e){switch(n){case L.PACKED_2X2_FLOAT32:return us(e);case L.PACKED_2X2_FLOAT16:return ds(e);case L.UNPACKED_FLOAT32:return is(e);case L.UNPACKED_FLOAT16:return cs(e);case L.PACKED_4X1_UNSIGNED_BYTE:return ls(e);default:throw new Error(`Unknown physical texture type ${n}`)}}function tf(n){return w().getBool("WEBGL_RENDER_FLOAT32_ENABLED")?n?L.PACKED_2X2_FLOAT32:L.UNPACKED_FLOAT32:n?L.PACKED_2X2_FLOAT16:L.UNPACKED_FLOAT16}function As(n,e){if(n===te.UPLOAD)return L.PACKED_2X2_FLOAT32;if(n===te.RENDER||n==null)return tf(e);if(n===te.DOWNLOAD||n===te.PIXELS)return L.PACKED_4X1_UNSIGNED_BYTE;throw new Error(`Unknown logical texture type ${n}`)}function Os(n,e,t){return`${n[0]}_${n[1]}_${e}_${t}`}class he{constructor(e,t){this.variableNames=["A"],this.outputShape=e,this.enableShapeUniforms=W(this.outputShape.length),this.userCode=`
      float unaryOperation(float x) {
        ${t}
      }

      void main() {
        float x = getAAtOutCoords();
        float y = unaryOperation(x);

        setOutput(y);
      }
    `}}const ie="if (isnan(x)) return x;",nf="return x;",Ds="return abs(x);",sf="return (x >= 0.0) ? x : (exp(x) - 1.0);",of=ie+`
  return (x < 0.0) ? 0.0 : x;
`,rf=ie+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,Ie="return x;",af="return 1.0 / (1.0 + exp(-1.0 * x));";const cf="return x;",lf=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,uf=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,df=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,pf="return 1.0 / (1.0 + exp(-1.0 * x));";class Re{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.enableShapeUniforms=W(this.outputShape.length),this.userCode=`
      vec4 unaryOperation(vec4 x) {
        ${t}
      }

      void main() {
        vec4 x = getAAtOutCoords();
        vec4 y = unaryOperation(x);

        setOutput(y);
      }
    `}}class hf{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outputShape=e,this.enableShapeUniforms=W(this.outputShape.length);const t=e.length,s=G("rc",t),o=_(t),r=qh(t,s),a=s.slice(-2),c=t<=1?"rc":`vec2(${a.join(",")})`;this.userCode=`
      void main() {
        ${o} rc = getOutputCoords();
        vec4 packedInput = getA(${r});

        setOutput(getChannel(packedInput, ${c}));
      }
    `}}const ff=ki,mf=1e-7,xf=1e-4,At={};function gf(n){return n in At||(At[n]={}),At[n]}const Cf=w().getNumber("CPU_HANDOFF_SIZE_THRESHOLD"),$f=600;function bf(){return w().global.screen==null?1024:w().global.screen.height*w().global.screen.width*window.devicePixelRatio*$f/1024/1024}class Rt extends Ti{nextDataId(){return Rt.nextDataId++}constructor(e){if(super(),this.pendingRead=new WeakMap,this.pendingDisposal=new WeakSet,this.dataRefCount=new WeakMap,this.numBytesInGPU=0,this.uploadWaitMs=0,this.downloadWaitMs=0,this.lastGlFlushTime=0,this.warnedAboutMemory=!1,this.pendingDeletes=0,this.disposed=!1,!w().getBool("HAS_WEBGL"))throw new Error("WebGL is not supported on this device");let t;if(e!=null){if(e instanceof Lt)t=e;else{const s=ue(w().getNumber("WEBGL_VERSION"),e);t=new Lt(s)}this.binaryCache={},this.gpgpuCreatedLocally=!1}else{const s=ue(w().getNumber("WEBGL_VERSION"));t=new Lt(s),this.binaryCache=gf(w().getNumber("WEBGL_VERSION")),this.gpgpuCreatedLocally=!0}this.gpgpu=t,this.canvas=this.gpgpu.gl.canvas,this.textureManager=new Zh(this.gpgpu),this.numMBBeforeWarning=bf(),this.texData=new Ei(this,we())}numDataIds(){return this.texData.numDataIds()-this.pendingDeletes}writeTexture(e,t,s,o,r,a){const c=this.makeTensorInfo(t,s),i=this.texData.get(c.dataId);i.isPacked=!1,i.texture={texture:e,texShape:[o,r]},i.texShape=[o,r];const l=ct(t),u=new Ns(l,!1,a),d=this.runWebGLProgram(u,[c],s,[[o,r]]);return d.shape=t,i.texture=null,this.disposeIntermediateTensorInfo(c),d.dataId}write(e,t,s){if((w().getBool("WEBGL_CHECK_NUMERICAL_PROBLEMS")||w().getBool("DEBUG"))&&this.checkNumericalProblems(e),s==="complex64"&&e!=null)throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");const o={id:this.nextDataId()};return this.texData.set(o,{shape:t,dtype:s,values:e,usage:te.UPLOAD,refCount:1}),o}refCount(e){return this.texData.has(e)?this.texData.get(e).refCount:0}incRef(e){const t=this.texData.get(e);t.refCount++}decRef(e){if(this.texData.has(e)){const t=this.texData.get(e);t.refCount--}}move(e,t,s,o,r){if(w().getBool("DEBUG")&&this.checkNumericalProblems(t),o==="complex64")throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");this.texData.set(e,{shape:s,dtype:o,values:t,usage:te.UPLOAD,refCount:r})}disposeIntermediateTensorInfo(e){this.disposeData(e.dataId)}readSync(e){const t=this.texData.get(e),{values:s,dtype:o,complexTensorInfos:r,slice:a,shape:c,isPacked:i}=t;if(a!=null){let p;i?p=new Re(c,Ie):p=new he(c,Ie);const h=this.runWebGLProgram(p,[{dataId:e,shape:c,dtype:o}],o),f=this.readSync(h.dataId);return this.disposeIntermediateTensorInfo(h),f}if(s!=null)return this.convertAndCacheOnCPU(e);if(o==="string")return s;const l=this.activeTimers!=null;let u;l&&(u=Ee());let d;if(o==="complex64"){const p=this.readSync(r.real.dataId),h=this.readSync(r.imag.dataId);d=pt(p,h)}else d=this.getValuesFromTexture(e);return l&&(this.downloadWaitMs+=Ee()-u),this.convertAndCacheOnCPU(e,d)}async read(e){if(this.pendingRead.has(e)){const f=this.pendingRead.get(e);return new Promise(g=>f.push(g))}const t=this.texData.get(e),{values:s,shape:o,slice:r,dtype:a,complexTensorInfos:c,isPacked:i}=t;if(r!=null){let f;i?f=new Re(o,Ie):f=new he(o,Ie);const g=this.runWebGLProgram(f,[{dataId:e,shape:o,dtype:a}],a),x=this.read(g.dataId);return this.disposeIntermediateTensorInfo(g),x}if(s!=null)return this.convertAndCacheOnCPU(e);if(w().getBool("DEBUG")&&!w().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")&&w().getNumber("WEBGL_VERSION")===2)throw new Error("tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and WEBGL_VERSION=2 not yet supported.");let l=null,u;if(a!=="complex64"&&w().get("WEBGL_BUFFER_SUPPORTED")){u=this.decode(e);const f=this.texData.get(u.dataId);l=this.gpgpu.createBufferFromTexture(f.texture.texture,...Nt(o))}this.pendingRead.set(e,[]),a!=="complex64"&&await this.gpgpu.createAndWaitForFence();let d;if(a==="complex64"){const f=await Promise.all([this.read(c.real.dataId),this.read(c.imag.dataId)]),g=f[0],x=f[1];d=pt(g,x)}else if(l==null)d=this.getValuesFromTexture(e);else{const f=E(o);d=this.gpgpu.downloadFloat32MatrixFromBuffer(l,f)}if(u!=null&&this.disposeIntermediateTensorInfo(u),l!=null){const f=this.gpgpu.gl;y(f,()=>f.deleteBuffer(l))}const p=this.convertAndCacheOnCPU(e,d),h=this.pendingRead.get(e);return this.pendingRead.delete(e),h.forEach(f=>f(p)),this.pendingDisposal.has(e)&&(this.pendingDisposal.delete(e),this.disposeData(e)&&we().removeDataId(e,this),this.pendingDeletes--),p}readToGPU(e,t={}){const s=this.texData.get(e),{values:o,shape:r,slice:a,dtype:c,isPacked:i,texture:l}=s;if(c==="complex64")throw new Error("Does not support reading texture for complex64 dtype.");if(a!=null){let h;i?h=new Re(r,Ie):h=new he(r,Ie);const f=this.runWebGLProgram(h,[{dataId:e,shape:r,dtype:c}],c),g=this.readToGPU(f,t);return this.disposeIntermediateTensorInfo(f),g}if(l==null)throw o!=null?new Error("Data is not on GPU but on CPU."):new Error("There is no data on GPU or CPU.");const u=this.decode(e,t.customTexShape),d=we().makeTensorFromTensorInfo(u),p=this.texData.get(u.dataId);return Object.assign({tensorRef:d},p.texture)}bufferSync(e){const t=this.readSync(e.dataId);if(e.dtype==="string")try{const s=t.map(o=>zt(o));return j(e.shape,e.dtype,s)}catch{throw new Error("Failed to decode encoded string bytes into utf-8")}return j(e.shape,e.dtype,t)}checkNumericalProblems(e){if(e!=null)for(let t=0;t<e.length;t++){const s=e[t];if(!_r(s))throw w().getBool("WEBGL_RENDER_FLOAT32_CAPABLE")?Error(`The value ${s} cannot be represented with your current settings. Consider enabling float32 rendering: 'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`):Error(`The value ${s} cannot be represented on this device.`)}}getValuesFromTexture(e){const{shape:t,dtype:s,isPacked:o}=this.texData.get(e),r=E(t);if(w().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")){const p=this.decode(e),h=this.texData.get(p.dataId),f=this.gpgpu.downloadMatrixFromPackedTexture(h.texture.texture,...Nt(t)).subarray(0,r);return this.disposeIntermediateTensorInfo(p),f}const a=w().getBool("WEBGL_PACK")&&o===!0,c=a?ct(t):t,i=a?new Jp(c):new Zp(c),l=this.runWebGLProgram(i,[{shape:c,dtype:s,dataId:e}],"float32"),u=this.texData.get(l.dataId),d=this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(u.texture.texture,u.texShape[0],u.texShape[1]).subarray(0,r);return this.disposeIntermediateTensorInfo(l),d}timerAvailable(){return w().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0}time(e){const t=this.activeTimers,s=[];let o=!1;this.programTimersStack==null?(this.programTimersStack=s,o=!0):this.activeTimers.push(s),this.activeTimers=s,e();const r=bs(this.activeTimers.map(i=>i.query)).filter(i=>i!=null),a=bs(this.activeTimers.map(i=>i.name)).filter(i=>i!=null);this.activeTimers=t,o&&(this.programTimersStack=null);const c={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null};return(async()=>{if(w().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0){const i=await Promise.all(r);c.kernelMs=Ni(i),c.getExtraProfileInfo=()=>i.map((l,u)=>({name:a[u],ms:l})).map(l=>`${l.name}: ${l.ms}`).join(", ")}else c.kernelMs={error:"WebGL query timers are not supported in this environment."};return this.uploadWaitMs=0,this.downloadWaitMs=0,c})()}memory(){return{unreliable:!1,numBytesInGPU:this.numBytesInGPU,numBytesInGPUAllocated:this.textureManager.numBytesAllocated,numBytesInGPUFree:this.textureManager.numBytesFree}}startTimer(){return w().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?this.gpgpu.beginQuery():{startMs:Ee(),endMs:null}}endTimer(e){return w().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?(this.gpgpu.endQuery(),e):(e.endMs=Ee(),e)}async getQueryTime(e){if(w().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0)return this.gpgpu.waitForQueryAndGetTime(e);const t=e;return t.endMs-t.startMs}disposeData(e,t=!1){if(this.pendingDisposal.has(e))return!1;if(!this.texData.has(e))return!0;if(t?this.texData.get(e).refCount=0:this.texData.get(e).refCount--,!t&&this.texData.get(e).refCount>0)return!1;if(this.pendingRead.has(e))return this.pendingDisposal.add(e),this.pendingDeletes++,!1;this.releaseGPUData(e);const{complexTensorInfos:s}=this.texData.get(e);return s!=null&&(this.disposeData(s.real.dataId,t),this.disposeData(s.imag.dataId,t)),this.texData.delete(e),!0}releaseGPUData(e){const{texture:t,dtype:s,texShape:o,usage:r,isPacked:a,slice:c}=this.texData.get(e),i=c&&c.origDataId||e,l=this.dataRefCount.get(i);l>1?this.dataRefCount.set(i,l-1):(this.dataRefCount.delete(i),t!=null&&(this.numBytesInGPU-=this.computeBytes(o,s),this.textureManager.releaseTexture(t,o,r,a)));const u=this.texData.get(e);u.texture=null,u.texShape=null,u.isPacked=!1,u.slice=null}getTexture(e){return this.uploadToGPU(e),this.texData.get(e).texture.texture}getDataInfo(e){return this.texData.get(e)}shouldExecuteOnCPU(e,t=Cf){return w().getBool("WEBGL_CPU_FORWARD")&&e.every(s=>this.texData.get(s.dataId).texture==null&&E(s.shape)<t)}getGPGPUContext(){return this.gpgpu}where(e){vt("tf.where() in webgl locks the UI thread. Call tf.whereAsync() instead");const t=e.dataSync();return ff(e.shape,t)}packedUnaryOp(e,t,s){const o=new Re(e.shape,t),r=this.compileAndRun(o,[e],s);return we().makeTensorFromTensorInfo(r)}abs(e){if(this.shouldExecuteOnCPU([e])&&e.dtype!=="complex64"){const o=Ea(this.texData.get(e.dataId).values);return this.makeOutput(e.shape,e.dtype,o)}if(w().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(e,Ds,e.dtype);const t=new he(e.shape,Ds),s=this.compileAndRun(t,[e]);return we().makeTensorFromTensorInfo(s)}makeTensorInfo(e,t,s){let o;if(t==="string"&&s!=null&&s.length>0&&Ai(s[0])){const r=s.map(a=>ut(a));o=this.write(r,e,t)}else o=this.write(s,e,t);return this.texData.get(o).usage=null,{dataId:o,shape:e,dtype:t}}makeOutput(e,t,s){return we().makeTensorFromTensorInfo(this.makeTensorInfo(e,t,s),this)}unpackTensor(e){const t=new hf(e.shape);return this.runWebGLProgram(t,[e],e.dtype)}packTensor(e){const t=new Yh(e.shape);return this.runWebGLProgram(t,[e],e.dtype,null,!0)}packedReshape(e,t){const s=[Oe(e.shape),...De(e.shape)],o={dtype:e.dtype,shape:s,dataId:e.dataId},r=[Oe(t),...De(t)],a=new Aa(r,s),c=!0,i=[s],l=this.runWebGLProgram(a,[o],e.dtype,i,c);return{dataId:l.dataId,shape:t,dtype:l.dtype}}decode(e,t){const s=this.texData.get(e),{isPacked:o,shape:r,dtype:a}=s;if(t!=null){const p=E(r),h=t[0]*t[1]*4;O(p<=h,()=>"customTexShape is too small. Row * Column * 4 should be equal or larger than the size of the tensor data.")}const c=ct(r);let i;o?i=new Qp(c):i=new Yp(c);const l=!0,u=[t??Nt(c)],d=this.runWebGLProgram(i,[{shape:c,dtype:a,dataId:e}],a,u,l,t);return{dtype:a,shape:r,dataId:d.dataId}}runWebGLProgram(e,t,s,o,r=!1,a){const c=this.makeTensorInfo(e.outputShape,s),i=this.texData.get(c.dataId);if(e.packedOutput&&(i.isPacked=!0),e.outPackingScheme===ft.DENSE){const m=a??Nt(e.outputShape);i.texShape=m.map(C=>C*2)}if(e.outTexUsage!=null&&(i.usage=e.outTexUsage),E(c.shape)===0)return i.values=ge(c.dtype,0),c;const l=[],u=t.map(m=>{if(m.dtype==="complex64")throw new Error("GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.");let C=this.texData.get(m.dataId);if(C.texture==null){if(!e.packedInputs&&E(m.shape)<=w().getNumber("WEBGL_SIZE_UPLOAD_UNIFORM"))return{shape:m.shape,texData:null,isUniform:!0,uniformValues:C.values};e.packedInputs&&(C.isPacked=!0,C.shape=m.shape)}if(this.uploadToGPU(m.dataId),!!C.isPacked!=!!e.packedInputs)m=C.isPacked?this.unpackTensor(m):this.packTensor(m),l.push(m),C=this.texData.get(m.dataId);else if(C.isPacked&&!mt(C.shape,m.shape)){const $=m,b=m.shape;m.shape=C.shape,m=this.packedReshape(m,b),l.push(m),C=this.texData.get(m.dataId),$.shape=b}return{shape:m.shape,texData:C,isUniform:!1}});this.uploadToGPU(c.dataId);const d={shape:c.shape,texData:i,isUniform:!1},p=qp(e,u,d),h=this.getAndSaveBinary(p,()=>Kp(this.gpgpu,e,u,d)),f=this.activeTimers!=null;let g;f&&(g=this.startTimer()),w().get("ENGINE_COMPILE_ONLY")||jp(this.gpgpu,h,u,d,o),l.forEach(m=>this.disposeIntermediateTensorInfo(m)),f&&(g=this.endTimer(g),this.activeTimers.push({name:e.constructor.name,query:this.getQueryTime(g)}));const x=w().getNumber("WEBGL_FLUSH_THRESHOLD");if(x>0){const m=Ee();m-this.lastGlFlushTime>x&&(this.gpgpu.gl.flush(),this.lastGlFlushTime=m)}if(!w().getBool("WEBGL_LAZILY_UNPACK")&&i.isPacked&&r===!1){const m=this.unpackTensor(c);return this.disposeIntermediateTensorInfo(c),m}return c}compileAndRun(e,t,s,o,r=!1){return s=s||t[0].dtype,this.runWebGLProgram(e,t,s,o,r)}getAndSaveBinary(e,t){return e in this.binaryCache||(this.binaryCache[e]=t()),this.binaryCache[e]}getTextureManager(){return this.textureManager}dispose(){this.disposed||(w().getBool("IS_TEST")||Object.keys(this.binaryCache).forEach(t=>{this.gpgpu.deleteProgram(this.binaryCache[t].webGLProgram),delete this.binaryCache[t]}),this.textureManager.dispose(),this.canvas!=null&&typeof HTMLCanvasElement<"u"&&this.canvas instanceof HTMLCanvasElement?this.canvas.remove():this.canvas=null,this.gpgpuCreatedLocally&&(this.gpgpu.program=null,this.gpgpu.dispose()),this.disposed=!0)}floatPrecision(){return this.floatPrecisionValue==null&&(this.floatPrecisionValue=co(()=>{if(!w().get("WEBGL_RENDER_FLOAT32_ENABLED")){const e=w().getBool("DEBUG");w().set("DEBUG",!1);const t=this.abs(Oi(1e-8)).dataSync()[0];if(w().set("DEBUG",e),t>0)return 32}return 16})),this.floatPrecisionValue}epsilon(){return this.floatPrecision()===32?mf:xf}uploadToGPU(e){const t=this.texData.get(e),{shape:s,dtype:o,values:r,texture:a,usage:c,isPacked:i}=t;if(a!=null)return;const l=this.activeTimers!=null;let u;l&&(u=Ee());let d=t.texShape;if(d==null&&(d=Jr(s,i),t.texShape=d),r!=null){const p=ct(s);let h,f=d[1],g=d[0];const x=r instanceof Uint8Array||r instanceof Uint8ClampedArray;(i||!x)&&([f,g]=qe(d[0],d[1])),i?h=new th(p,x):h=new Ns(p,x);const m=x?[g,f]:d,C=this.makeTensorInfo(m,o),$=this.texData.get(C.dataId);x?$.usage=te.PIXELS:$.usage=te.UPLOAD,$.texShape=m,this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(C.dataId),f,g,r);const b=[[g,f]],T=this.runWebGLProgram(h,[C],o,b,!0),S=this.texData.get(T.dataId);t.texShape=S.texShape,t.isPacked=S.isPacked,t.usage=S.usage,w().get("ENGINE_COMPILE_ONLY")?this.disposeData(T.dataId):(t.texture=S.texture,t.values=null,this.texData.delete(T.dataId)),this.disposeIntermediateTensorInfo(C),l&&(this.uploadWaitMs+=Ee()-u)}else{const p=this.acquireTexture(d,c,o,i);t.texture=p}}convertAndCacheOnCPU(e,t){const s=this.texData.get(e),{dtype:o}=s;return t!=null&&(s.values=vf(t,o)),s.values}acquireTexture(e,t,s,o){if(this.numBytesInGPU+=this.computeBytes(e,s),!this.warnedAboutMemory&&this.numBytesInGPU>this.numMBBeforeWarning*1024*1024){const r=(this.numBytesInGPU/1024/1024).toFixed(2);this.warnedAboutMemory=!0,console.warn(`High memory usage in GPU: ${r} MB, most likely due to a memory leak`)}return this.textureManager.acquireTexture(e,t,o)}computeBytes(e,t){return e[0]*e[1]*Di(t)}checkCompileCompletion(){for(const[,e]of Object.entries(this.binaryCache))this.checkCompletion_(e)}async checkCompileCompletionAsync(){const e=[];if(this.gpgpu.parallelCompilationExtension){for(const[,t]of Object.entries(this.binaryCache))e.push(this.checkCompletionAsync_(t));return Promise.all(e)}else{for(const[,t]of Object.entries(this.binaryCache)){const s=new Promise(o=>{try{this.checkCompletion_(t),o(!0)}catch(r){throw r}});e.push(s)}return Promise.all(e)}}async checkCompletionAsync_(e){return this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.parallelCompilationExtension.COMPLETION_STATUS_KHR)?this.checkCompletion_(e):(await Tu(),this.checkCompletionAsync_(e))}checkCompletion_(e){if(this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.gl.LINK_STATUS)===!1)throw console.log(this.gpgpu.gl.getProgramInfoLog(e.webGLProgram)),this.gpgpu.gl.getShaderParameter(e.fragmentShader,this.gpgpu.gl.COMPILE_STATUS)===!1?(ss(e.source,this.gpgpu.gl.getShaderInfoLog(e.fragmentShader)),new Error("Failed to compile fragment shader.")):new Error("Failed to link vertex and fragment shaders.");return!0}getUniformLocations(){for(const e of Object.values(this.binaryCache)){this.gpgpu.buildVao(e.webGLProgram);const{variablesLocations:t,customUniformLocations:s,infLoc:o,nanLoc:r,outShapeLocation:a,outShapeStridesLocation:c,outTexShapeLocation:i}=ua(this.gpgpu,e.program,e.webGLProgram);e.variablesLocations=t,e.customUniformLocations=s,e.infLoc=o,e.nanLoc=r,e.outShapeLocation=a,e.outShapeStridesLocation=c,e.outTexShapeLocation=i}}createTensorFromGPUData(e,t,s){e.channels=e.channels||"RGBA";const{texture:o,height:r,width:a,channels:c}=e,i=we().backend;if(!i.gpgpu.gl.isTexture(o))throw new Error("The texture is invalid. Also, please make sure the texture and the TFJS WebGL backend are using the same canvas. If you want to use your own custom canvas, you have to create and use the custom TFJS WebGL backend created from the canvas through 'new tf.MathBackendWebGL(customCanvas)'.");const l=i.writeTexture(o,t,s,r,a,c);return we().makeTensorFromDataId(l,t,s,i)}}Rt.nextDataId=0;function vf(n,e){if(e==="float32"||e==="complex64")return n;if(e==="int32"||e==="bool"){const t=e==="int32"?new Int32Array(n.length):new Uint8Array(n.length);for(let s=0;s<t.length;++s)t[s]=Math.round(n[s]);return t}else throw new Error(`Unknown dtype ${e}`)}const wf="4.22.0";function Oa(){w().set("WEBGL_FORCE_F16_TEXTURES",!0)}Fi()&&Pi("webgl",()=>new Rt,2);const If={forceHalfFloat:Oa};const hs=`
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`;class Fe{constructor(e,t,s){this.variableNames=["A","B"],this.outputShape=z(t,s),this.enableShapeUniforms=W(this.outputShape.length),this.userCode=`
      float binaryOperation(float a, float b) {
        ${e}
      }

      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `}}const Be=`
  result.r = isNaN.r ? NAN : result.r;
  result.g = isNaN.g ? NAN : result.g;
  result.b = isNaN.b ? NAN : result.b;
  result.a = isNaN.a ? NAN : result.a;
`;class tt{constructor(e,t,s,o=!1){this.variableNames=["A","B"],this.supportsBroadcasting=!0,this.packedInputs=!0,this.packedOutput=!0,this.outputShape=z(t,s);const r=this.outputShape.length;this.enableShapeUniforms=W(r);let a="";if(o)if(r===0||E(this.outputShape)===1)a=`
          result.y = 0.;
          result.z = 0.;
          result.w = 0.;
        `;else if(a=`
          ${_(r)} coords = getOutputCoords();
        `,r===1)this.enableShapeUniforms?a+=`
            result.y = (coords + 1) >= outShape ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `:a+=`
            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;else{const i=G("coords",r);this.enableShapeUniforms?a+=`
            bool nextRowOutOfBounds =
              (${i[r-2]} + 1) >= outShape[${r} - 2];
            bool nextColOutOfBounds =
              (${i[r-1]} + 1) >= outShape[${r} - 1];
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `:a+=`
            bool nextRowOutOfBounds =
              (${i[r-2]} + 1) >= ${this.outputShape[r-2]};
            bool nextColOutOfBounds =
              (${i[r-1]} + 1) >= ${this.outputShape[r-1]};
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `}this.userCode=`
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${e}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();

        vec4 result = binaryOperation(a, b);
        ${a}

        setOutput(result);
      }
    `}}function Z(n){const{inputs:e,backend:t}=n,{x:s}=e;return t.incRef(s.dataId),{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}const Rf={kernelName:to,backendName:"webgl",kernelFunc:Z};function Te(n){const{inputs:e,backend:t}=n,{real:s,imag:o}=e,r=t.makeTensorInfo(s.shape,"complex64"),a=t.texData.get(r.dataId),c=Z({inputs:{x:s},backend:t}),i=Z({inputs:{x:o},backend:t});return a.complexTensorInfos={real:c,imag:i},r}const yf={kernelName:eo,backendName:"webgl",kernelFunc:Te};const Da="return (a < 0.) ? b * a : a;",Fa=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function Sf(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{alpha:r}=s,a=t.makeTensorInfo([],"float32",je(r,"float32")),c=w().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new tt(Fa,o.shape,a.shape):new Fe(Da,o.shape,a.shape),i=t.runWebGLProgram(c,[o,a],"float32");return t.disposeIntermediateTensorInfo(a),i}const Tf={kernelName:_i,backendName:"webgl",kernelFunc:Sf};const Pa="return (a < 0.) ? b * a : a;",_a=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function Ef(n){const{inputs:e,backend:t}=n,{x:s,alpha:o}=e,r=w().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new tt(_a,s.shape,o.shape):new Fe(Pa,s.shape,o.shape);return t.runWebGLProgram(r,[s,o],"float32")}const Nf={kernelName:Li,backendName:"webgl",kernelFunc:Ef};const nt="if (isnan(x)) return x;";function D({opSnippet:n,packedOpSnippet:e,cpuKernelImpl:t,dtype:s}){return({inputs:o,backend:r})=>{const{x:a}=o,c=r,i=s||a.dtype;if(c.shouldExecuteOnCPU([a])&&t!=null){const d=c.texData.get(a.dataId),p=t(d.values,i);return c.makeTensorInfo(a.shape,i,p)}const l=w().getBool("WEBGL_PACK_UNARY_OPERATIONS")&&e!=null;let u;return l?u=new Re(a.shape,e):u=new he(a.shape,n),c.runWebGLProgram(u,[a],i)}}function V({opSnippet:n,packedOpSnippet:e,checkOutOfBounds:t=!1,supportsComplex:s=!1,cpuKernelImpl:o,dtype:r}){return({inputs:a,backend:c})=>{const{a:i,b:l}=a,u=c;if(s&&i.dtype==="complex64"){const f=u.texData.get(i.dataId),g=u.texData.get(l.dataId),[x,m]=[[f.complexTensorInfos.real,g.complexTensorInfos.real],[f.complexTensorInfos.imag,g.complexTensorInfos.imag]].map($=>{const[b,v]=$,T={dataId:b.dataId,dtype:b.dtype,shape:i.shape},S={dataId:v.dataId,dtype:v.dtype,shape:l.shape},I=new Fe(n,i.shape,l.shape);return u.runWebGLProgram(I,[T,S],ye(b.dtype,v.dtype))}),C=Te({inputs:{real:x,imag:m},backend:u});return u.disposeIntermediateTensorInfo(x),u.disposeIntermediateTensorInfo(m),C}const d=r||ye(i.dtype,l.dtype);if((i.dtype==="string"||l.dtype==="string"||u.shouldExecuteOnCPU([i,l]))&&o!=null){const f=u.texData.get(i.dataId).values,g=u.texData.get(l.dataId).values,x=i.dtype==="string"?Ce(f):f,m=i.dtype==="string"?Ce(g):g,[C,$]=o(i.shape,l.shape,x,m,d),b=u.makeTensorInfo($,d),v=u.texData.get(b.dataId);return v.values=C,b}const p=w().getBool("WEBGL_PACK_BINARY_OPERATIONS")&&e!=null;let h;return p?h=new tt(e,i.shape,l.shape,t):h=new Fe(n,i.shape,l.shape),u.runWebGLProgram(h,[i,l],d)}}function xt(n,e=!1){if(n==="linear")return e?cf:nf;if(n==="relu")return e?uf:of;if(n==="elu")return e?lf:sf;if(n==="relu6")return e?df:rf;if(n==="prelu")return e?_a:Pa;if(n==="leakyrelu")return e?Fa:Da;if(n==="sigmoid")return e?pf:af;throw new Error(`Activation ${n} has not been implemented for the WebGL backend.`)}class La{constructor(e,t,s,o=!1,r=!1,a=!1,c=null,i=!1,l=!1){this.variableNames=["matrixA","matrixB"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=s,this.enableShapeUniforms=W(this.outputShape.length);const u=o?e[1]:e[2],d=Math.ceil(u/2),p=o?"i * 2, rc.y":"rc.y, i * 2",h=r?"rc.z, i * 2":"i * 2, rc.z",f=o?["a.xxyy","a.zzww"]:["a.xxzz","a.yyww"],g=r?["b.xzxz","b.ywyw"]:["b.xyxy","b.zwzw"];let x="",m="";c&&(i?x=`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${c}
        }`:l?x=`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${c}
        }`:x=`vec4 activation(vec4 x) {
          ${c}
        }`,m="result = activation(result);");const C=a?"result += getBiasAtOutCoords();":"";a&&this.variableNames.push("bias"),i&&this.variableNames.push("preluActivationWeights"),l&&this.variableNames.push("leakyreluAlpha");let $="rc.x",b="rc.x";e[0]<t[0]?$=`imod(rc.x, ${e[0]})`:t[0]<e[0]&&(b=`imod(rc.x, ${t[0]})`),this.userCode=`
      ${x}
      // Don't use uniform for sharedDimensionPacked for performance.
      const float sharedDimension = ${d}.0;

      vec4 dot2x2ARowBCol(ivec3 rc) {
        vec4 result = vec4(0);
        int batchA = ${$};
        int batchB = ${b};
        for (int i = 0; i < ${d}; i++) {
          vec4 a = getMatrixA(batchA, ${p});
          vec4 b = getMatrixB(batchB, ${h});

          // These swizzled products need to be separately added.
          // See: https://github.com/tensorflow/tfjs/issues/1735
          result += (${f[0]} * ${g[0]});
          result += (${f[1]} * ${g[1]});
        }
        return result;
      }

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = dot2x2ARowBCol(rc);

        ${C}

        ${m}

        setOutput(result);
      }
    `}}const Fs={REAL:"return areal * breal - aimag * bimag;",IMAG:"return areal * bimag + aimag * breal;"};class Ps{constructor(e,t,s){this.variableNames=["AReal","AImag","BReal","BImag"],this.outputShape=z(t,s),this.userCode=`
      float binaryOpComplex(
          float areal, float aimag, float breal, float bimag) {
        ${e}
      }

      void main() {
        float areal = getARealAtOutCoords();
        float aimag = getAImagAtOutCoords();
        float breal = getBRealAtOutCoords();
        float bimag = getBImagAtOutCoords();
        setOutput(binaryOpComplex(areal, aimag, breal, bimag));
      }
    `}}const _s="return a * b;";function fs(n){const{inputs:e,backend:t}=n,{a:s,b:o}=e,r=ye(s.dtype,o.dtype);if(s.dtype==="complex64"){const c=t.texData.get(s.dataId),i=t.texData.get(o.dataId),l=new Ps(Fs.REAL,s.shape,o.shape),u=new Ps(Fs.IMAG,s.shape,o.shape),d=[{dataId:c.complexTensorInfos.real.dataId,dtype:c.complexTensorInfos.real.dtype,shape:s.shape},{dataId:c.complexTensorInfos.imag.dataId,dtype:c.complexTensorInfos.imag.dtype,shape:s.shape},{dataId:i.complexTensorInfos.real.dataId,dtype:i.complexTensorInfos.real.dtype,shape:o.shape},{dataId:i.complexTensorInfos.imag.dataId,dtype:i.complexTensorInfos.imag.dtype,shape:o.shape}],p=t.runWebGLProgram(l,d,"float32"),h=t.runWebGLProgram(u,d,"float32"),f=Te({inputs:{real:p,imag:h},backend:t});return t.disposeIntermediateTensorInfo(p),t.disposeIntermediateTensorInfo(h),f}if(t.shouldExecuteOnCPU([s,o])){const c=t.texData.get(s.dataId),i=t.texData.get(o.dataId),[l,u]=yh(s.shape,o.shape,c.values,i.values,r),d=t.makeTensorInfo(u,r),p=t.texData.get(d.dataId);return p.values=l,d}let a;return w().getBool("WEBGL_PACK_BINARY_OPERATIONS")?a=new tt(_s,s.shape,o.shape):a=new Fe(_s,s.shape,o.shape),t.runWebGLProgram(a,[s,o],r)}const kf={kernelName:Pn,backendName:"webgl",kernelFunc:fs};function Af(n,e,t){const s=[Oe(n.shape),...De(n.shape)],o={dtype:n.dtype,shape:s,dataId:n.dataId},r=[Oe(e),...De(e)],a=new Aa(r,s),c=!0,i=[s],l=t.runWebGLProgram(a,[o],n.dtype,i,c);return{dataId:l.dataId,shape:e,dtype:l.dtype}}function R(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{shape:r}=s,a=t,c=E(o.shape),i=Bi(r,c),l=E(i);O(c===l,()=>`The new shape (${i}) has ${l} elements and the old shape (${o.shape}) has ${c} elements. The new shape and old shape must have the same number of elements.`);const u=a.texData.get(o.dataId);return u.isPacked&&!mt(o.shape,i)&&!(u.texture!==null&&mt(u.shape,i))?Af(o,i,a):(a.incRef(o.dataId),{dataId:o.dataId,shape:i,dtype:o.dtype})}const Of={kernelName:Vi,backendName:"webgl",kernelFunc:R};class Ls{constructor(e,t){this.variableNames=["x"];const{windowSize:s,batchSize:o,inSize:r,outSize:a}=e;this.outputShape=[o,a];const c=Math.floor(s/4)*4,i=s%4;let l="sumValue += dot(values, ones);";if(t!=null){const d=1/t;l=`sumValue += dot(values * ${Mi(d)?d.toPrecision(2):d}, ones);`}let u="";r%s>0&&(u=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return 0.0;
        }
      `),this.userCode=`
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${u}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${s};

        float sumValue = 0.0;

        for (int i = 0; i < ${c}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${l}
        }

        int inIdx = inOffset + ${c};
        if (${i===1}) {
          vec4 values = vec4(getValue(batch, inIdx), 0.0, 0.0, 0.0);

          ${l}
        } else if (${i===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1), 0.0, 0.0);

          ${l}
        } else if (${i===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2), 0.0);

          ${l}
        }
        setOutput(sumValue);
      }
    `}}class Df{constructor(e,t){this.variableNames=["x"];const{windowSize:s,batchSize:o,inSize:r,outSize:a}=e;this.outputShape=[o,a];let c="0.0",i="";t==="prod"?c="1.0":t==="min"?(c="1.0 / 1e-20",i="min"):t==="max"&&(c="-1.0 / 1e-20",i="max");let l=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t==="sum"?l="sumValue":t==="prod"?l="prodValue":t==="all"?l="allValue":t==="any"&&(l="anyValue");const u=Math.floor(s/4)*4,d=s%4;let p=`
      if (${t==="sum"}) {
        sumValue += dot(values, ones);
      } else if (${t==="prod"}) {
        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);
        prodValue *= tmp[0] * tmp[1];
      } else {
        minMaxValue = ${i}(values, minMaxValue);
        if (${t==="min"} || ${t==="max"}) {
          minMaxValue = ${i}(values, minMaxValue);
          bvec4 isNaN = isnan(values);
          if (isNaN.r || isNaN.g || isNaN.b || isNaN.a) {
            minMaxValue = vec4(NAN);
          }
        }
      }
    `,h="vec4";t==="all"?(c="1.0",p=`
        bool reducedAllValue = all(values);
        float floatedReducedAllValue = float(reducedAllValue);
        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);
      `,h="bvec4"):t==="any"&&(c="0.0",p=`
        bool reducedAnyValue = any(values);
        float floatedReducedAnyValue = float(reducedAnyValue);
        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);
      `,h="bvec4");let f="";r%s>0&&(f=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return initializationValue;
        }
      `),this.userCode=`
      const float initializationValue = ${c};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${f}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${s};

        vec4 minMaxValue = vec4(${c});
        float prodValue = 1.0;
        float sumValue = 0.0;
        float allValue = 1.0;
        float anyValue = 0.0;

        for (int i = 0; i < ${u}; i += 4) {
          int inIdx = inOffset + i;
          ${h} values = ${h}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${p}
        }

        int inIdx = inOffset + ${u};
        if (${d===1}) {
          ${h} values = ${h}(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          ${p}
        } else if (${d===2}) {
          ${h} values = ${h}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          ${p}
        } else if (${d===3}) {
          ${h} values = ${h}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          ${p}
        }
        setOutput(${l});
      }
    `}}function Ff(n){const e=[];for(;e.length===0||e[e.length-1].outSize!==1;){const t=e.length?e[e.length-1].outSize:n[1],s=jt(t);e.push({inSize:t,windowSize:s,outSize:Math.ceil(t/s)})}return e}function Me(n,e,t,s){const o=Ff(n.shape);let r=n;for(let a=0;a<o.length;a++){const{inSize:c,windowSize:i,outSize:l}=o[a];let u,d;t==="mean"?u=a===0?new Ls({windowSize:i,inSize:c,batchSize:n.shape[0],outSize:l},c):new Ls({windowSize:i,inSize:c,batchSize:n.shape[0],outSize:l}):u=new Df({windowSize:i,inSize:c,batchSize:n.shape[0],outSize:l},t),d=r,r=s.runWebGLProgram(u,[r],e),d.dataId!==n.dataId&&s.disposeIntermediateTensorInfo(d)}return r}class Pf{constructor(e,t){this.variableNames=["A"];const s=new Array(e.length);for(let a=0;a<s.length;a++)s[a]=e[t[a]];this.outputShape=s,this.rank=s.length;const o=_(this.rank),r=_f(t);this.userCode=`
    void main() {
      ${o} resRC = getOutputCoords();
      setOutput(getA(${r}));
    }
    `}}function _f(n){const e=n.length;if(e>6)throw Error(`Transpose for rank ${e} is not yet supported`);const t=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u","resRC.v"],s=new Array(e);for(let o=0;o<n.length;o++)s[n[o]]=t[o];return s.join()}class Lf{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0;const s=new Array(e.length);for(let u=0;u<s.length;u++)s[u]=e[t[u]];if(this.outputShape=s,this.rank=s.length,this.rank>6)throw Error(`Packed transpose for rank ${this.rank} is not yet supported.`);const o=_(this.rank),r=ka("rc",this.rank),a=new Array(this.rank);for(let u=0;u<t.length;u++)a[t[u]]=r[u];const c=`vec2(${a.slice(-2).join()})`,i=`++${r[this.rank-1]} < ${s[this.rank-1]}`,l=`getChannel(getA(${a.join()}), ${c})`;this.userCode=`
    void main() {
      ${o} rc = getOutputCoords();
      vec4 result = vec4(0.);
      result[0] = ${l};
      if(${i}) {
        result[1] = ${l};
      }
      --${r[this.rank-1]};
      if(++${r[this.rank-2]} < ${s[this.rank-2]}) {
        result[2] = ${l};
        if(${i}) {
          result[3] = ${l};
        }
      }
      setOutput(result);
    }
    `}}function Qt(n,e,t){const s=w().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new Lf(n.shape,e):new Pf(n.shape,e);return t.runWebGLProgram(s,[n],n.dtype)}function Vf(n,e,t,s){const o=e,r=n.shape.length,a=X(o,n.shape);let c=a;const i=se(c,r),l=i!=null;let u=n;l&&(u=Qt(n,i,s),c=oe(c.length,r)),de("sum",c,r);const[d,p]=fe(u.shape,c);let h=d;t&&(h=me(d,a));const f=E(p),x=E(n.shape)/f,m=R({inputs:{x:u},attrs:{shape:[x,f]},backend:s}),C=zn(n.dtype),$=Me(m,C,"sum",s),b=R({inputs:{x:$},attrs:{shape:h},backend:s});return s.disposeIntermediateTensorInfo(m),s.disposeIntermediateTensorInfo($),l&&s.disposeIntermediateTensorInfo(u),b}function Zt(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,keepDims:a}=s;return Vf(o,r,a,t)}const Bf={kernelName:Wi,backendName:"webgl",kernelFunc:Zt};function H(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{perm:r}=s,a=t,c=o.shape.length,i=new Array(c);for(let u=0;u<i.length;u++)i[u]=o.shape[r[u]];let l;if(a.shouldExecuteOnCPU([o])){const d=a.texData.get(o.dataId).values,p=ps(d,o.shape,o.dtype,r,i);l=a.makeTensorInfo(i,o.dtype);const h=a.texData.get(l.dataId);h.values=p}else l=Qt(o,r,a);return l}const Mf={kernelName:ao,backendName:"webgl",kernelFunc:H};const Va=1e3;function Wt({a:n,b:e,transposeA:t,transposeB:s,backend:o,bias:r=null,preluActivationWeights:a=null,leakyreluAlpha:c=0,activation:i=null}){const l=n.shape.length,u=e.shape.length,d=t?n.shape[l-2]:n.shape[l-1],p=s?e.shape[u-1]:e.shape[u-2],h=t?n.shape[l-1]:n.shape[l-2],f=s?e.shape[u-2]:e.shape[u-1],g=n.shape.slice(0,-2),x=e.shape.slice(0,-2),m=E(g),C=E(x),b=z(n.shape.slice(0,-2),e.shape.slice(0,-2)).concat([h,f]);O(d===p,()=>`Error in matMul: inner shapes (${d}) and (${p}) of Tensors with shapes ${n.shape} and ${e.shape} and transposeA=${t} and transposeB=${s} must match.`);const v=t?[m,d,h]:[m,h,d],T=s?[C,f,p]:[C,p,f],S=R({inputs:{x:n},backend:o,attrs:{shape:v}}),I=R({inputs:{x:e},backend:o,attrs:{shape:T}}),A=[S,I],k=Math.max(m,C),F=t?S.shape[1]:S.shape[2],P=r!=null,pe=a!=null,Q=i==="leakyrelu",ee=i!=null?xt(i,!0):null,re=P||pe||Q||ee!=null;let ce;if((h===1||f===1)&&F>Va&&re===!1){let ve=S,We=I;t&&(ve=H({inputs:{x:S},backend:o,attrs:{perm:[0,2,1]}}),A.push(ve)),s&&(We=H({inputs:{x:I},backend:o,attrs:{perm:[0,2,1]}}),A.push(We));const Ue=f!==1,Tt=f===1;let en=ve;Ue&&(en=R({inputs:{x:ve},backend:o,attrs:{shape:[k,F,1]}}),A.push(en));const ci=f===1?2:1;let tn=We;Tt&&(tn=R({inputs:{x:We},backend:o,attrs:{shape:[k,1,F]}}),A.push(tn));const gs=fs({inputs:{a:en,b:tn},backend:o});ce=Zt({inputs:{x:gs},backend:o,attrs:{axis:ci,keepDims:!0}}),A.push(gs)}else{const ve=ye(n.dtype,e.dtype),We=new La(v,T,[k,h,f],t,s,P,ee,pe,Q),Ue=[S,I];if(r!=null&&Ue.push(r),pe&&Ue.push(a),Q){const Tt=o.makeTensorInfo([],"float32",je(c,"float32"));Ue.push(Tt),A.push(Tt)}ce=o.runWebGLProgram(We,Ue,ve)}const U=R({inputs:{x:ce},backend:o,attrs:{shape:b}});A.push(ce);for(const ve of A)o.disposeIntermediateTensorInfo(ve);return U}function Wf(n){const{inputs:e,backend:t,attrs:s}=n,{a:o,b:r,bias:a,preluActivationWeights:c}=e,{transposeA:i,transposeB:l,activation:u,leakyreluAlpha:d}=s;return Wt({a:o,b:r,transposeA:i,transposeB:l,backend:t,bias:a,preluActivationWeights:c,leakyreluAlpha:d,activation:u})}const Uf={kernelName:Ui,backendName:"webgl",kernelFunc:Wf};const Vs="return abs(x);";function Gf(n){const{inputs:e,backend:t}=n,{x:s}=e;if(t.shouldExecuteOnCPU([s])&&s.dtype!=="complex64"){const r=t.texData.get(s.dataId),a=Ea(r.values);return t.makeTensorInfo(s.shape,s.dtype,a)}let o;return w().getBool("WEBGL_PACK_UNARY_OPERATIONS")?o=new Re(s.shape,Vs):o=new he(s.shape,Vs),t.runWebGLProgram(o,[s],s.dtype)}const zf={kernelName:Js,backendName:"webgl",kernelFunc:Gf};const Hf=ie+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return acos(x);
`,Xf=D({opSnippet:Hf}),Kf={kernelName:Gi,backendName:"webgl",kernelFunc:Xf};const jf=ie+`
  if (x < 1.0) return NAN;
return log(x + sqrt(x * x - 1.0));`,qf=D({opSnippet:jf}),Yf={kernelName:zi,backendName:"webgl",kernelFunc:qf};const Bs="return a + b;",Qf=V({opSnippet:Bs,packedOpSnippet:Bs,supportsComplex:!0,cpuKernelImpl:oh}),Zf={kernelName:bn,backendName:"webgl",kernelFunc:Qf};class Jf{constructor(e,t){this.outputShape=[],this.outputShape=e,this.variableNames=t.map((r,a)=>`T${a}`);const s=[];this.variableNames.forEach(r=>{s.push(`float v${r} = get${r}AtOutCoords();`)});const o=this.variableNames.map(r=>`v${r}`).join(" + ");this.userCode=`
      void main() {
        ${s.join(`
        `)}

        float result = ${o};
        setOutput(result);
      }
    `}}class em{constructor(e,t){this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.variableNames=t.map((r,a)=>`T${a}`);const s=[];this.variableNames.forEach(r=>{s.push(`vec4 v${r} = get${r}AtOutCoords();`)});const o=this.variableNames.map(r=>`v${r}`).join(" + ");this.userCode=`
      void main() {
        ${s.join(`
        `)}

        vec4 result = ${o};
        setOutput(result);
      }
    `}}function Vt(n){const{inputs:e,backend:t}=n,s=e;if(s.length===1)return Z({inputs:{x:s[0]},backend:t});if(s.length>w().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER")){const i=Math.floor(s.length/2),l=Vt({inputs:s.slice(0,i),backend:t}),u=Vt({inputs:s.slice(i),backend:t});return Vt({inputs:[l,u],backend:t})}const o=s.map(i=>i.dtype).reduce((i,l)=>ye(i,l)),r=s.map(i=>i.shape),c=w().getBool("WEBGL_PACK")?new em(s[0].shape,r):new Jf(s[0].shape,r);return t.runWebGLProgram(c,s,o)}const tm={kernelName:Hi,backendName:"webgl",kernelFunc:Vt};function nm(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,keepDims:a}=s,c=o.shape.length,i=X(r,o.shape);let l=i;const u=se(l,c);let d=o;u!=null&&(d=H({inputs:{x:o},backend:t,attrs:{perm:u}}),l=oe(l.length,c)),de("all",l,c);const[p,h]=fe(d.shape,l),f=E(h),g=R({inputs:{x:d},backend:t,attrs:{shape:[-1,f]}}),x=Me(g,g.dtype,"all",t);let m;if(a){const C=me(p,i);m=R({inputs:{x},backend:t,attrs:{shape:C}})}else m=R({inputs:{x},backend:t,attrs:{shape:p}});return t.disposeIntermediateTensorInfo(g),t.disposeIntermediateTensorInfo(x),u!=null&&t.disposeIntermediateTensorInfo(d),m}const sm={kernelName:Xi,backendName:"webgl",kernelFunc:nm};function om(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,keepDims:a}=s,c=o.shape.length,i=X(r,o.shape);let l=i;const u=se(l,c);let d=o;u!=null&&(d=H({inputs:{x:o},backend:t,attrs:{perm:u}}),l=oe(l.length,c)),de("any",l,c);const[p,h]=fe(d.shape,l),f=E(h),g=R({inputs:{x:d},backend:t,attrs:{shape:[-1,f]}}),x=Me(g,g.dtype,"any",t);let m;if(a){const C=me(p,i);m=R({inputs:{x},backend:t,attrs:{shape:C}})}else m=R({inputs:{x},backend:t,attrs:{shape:p}});return t.disposeIntermediateTensorInfo(g),t.disposeIntermediateTensorInfo(x),u!=null&&t.disposeIntermediateTensorInfo(d),m}const rm={kernelName:Ki,backendName:"webgl",kernelFunc:om};class am{constructor(e,t,s){this.variableNames=["A"];const{windowSize:o,batchSize:r,outSize:a}=e;s||this.variableNames.push("bestIndicesA"),this.outputShape=[r,a];const c=t==="max"?">":"<",i=s?"inOffset + i;":"round(getBestIndicesA(batch, inOffset + i));";this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${o};

        int bestIndex = inOffset;
        float bestValue = getA(batch, bestIndex);

        for (int i = 0; i < ${o}; i++) {
          int inIdx = ${i};
          float candidate = getA(batch, inIdx);
          if (candidate ${c} bestValue) {
            bestValue = candidate;
            bestIndex = inIdx;
          }
        }
        setOutput(float(bestIndex));
      }
    `}}class im{constructor(e,t,s,o){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,O(e.length>2,()=>`Packed arg${s.charAt(0).toUpperCase()+s.slice(1)} supports only inputs with rank above 2.`);const r=e[e.length-1],a=Math.ceil(r/t);this.outputShape=e.slice(0,-1),a>1&&this.outputShape.push(a),o||this.variableNames.push("bestIndicesA");const c=this.outputShape,i=c.length,l=_(i),u=G("coords",i);let d,p;if(a===1){p=i+1;const I=_(p);d=`
        ${I} sourceLocR = ${I}(${u.join()}, 0);
        ++${u[i-1]};
        ${I} sourceLocG = ${I}(${u.join()}, 0);
        ++${u[i-2]};
        ${I} sourceLocA = ${I}(${u.join()}, 0);
        --${u[i-1]};
        ${I} sourceLocB = ${I}(${u.join()}, 0);
        --${u[i-2]};`}else p=i,d=`
        ${l} sourceLocR = coords;
        ++${u[i-1]};
        ${l} sourceLocG = coords;
        ++${u[i-2]};
        ${l} sourceLocA = coords;
        --${u[i-1]};
        ${l} sourceLocB = coords;
        --${u[i-2]};`;const h=["x","y","z","w","u","v"].slice(0,p),f="."+h[p-1],g=h.map(I=>"int "+I),x=G("sourceLocR",p-1).concat("inIdx.r"),m=G("sourceLocG",p-1).concat("inIdx.g"),C=G("sourceLocB",p-1).concat("inIdx.b"),$=G("sourceLocA",p-1).concat("inIdx.a"),b=s==="max"?"greaterThan":"lessThan",v=o?"":`
          inIdx = round(vec4(getBestIndicesAChannel(${x.join()}),
                             getBestIndicesAChannel(${m.join()}),
                             getBestIndicesAChannel(${C.join()}),
                             getBestIndicesAChannel(${$.join()})));`,T=`vec4(
            getAChannel(${x.join()}),
            hasNextCol ? getAChannel(${m.join()}) : 0.,
            hasNextRow ? getAChannel(${C.join()}) : 0.,
            hasNextRow && hasNextCol ? getAChannel(${$.join()}) : 0.)`,S=o?"":`
      float getBestIndicesAChannel(${g.join()}) {
        return getChannel(getBestIndicesA(${h.join()}),
                                          vec2(${h.slice(-2).join()}));
      }`;this.userCode=`
      float getAChannel(${g.join()}) {
        return getChannel(getA(${h.join()}),
                               vec2(${h.slice(-2).join()}));
      }
      ${S}
      void main() {
        ${l} coords = getOutputCoords();
        bool hasNextCol = ${u[i-1]} < ${c[i-1]-1};
        bool hasNextRow = ${u[i-2]} < ${c[i-2]-1};
        ${d}
        ivec4 srcIdx = ivec4(sourceLocR${f}, sourceLocG${f},
          sourceLocB${f}, sourceLocA${f}) * ${t};
        ivec4 inIdx = srcIdx;
        vec4 bestIndex = vec4(inIdx);
        vec4 bestValue = ${T};

        for (int i = 0; i < ${t}; i++) {
          inIdx = srcIdx;
          ${v}
          vec4 candidate = ${T};
          bvec4 nan = isnan(candidate);
          bvec4 replace = bvec4(
            vec4(${b}(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));

          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,
                           replace.y  ? candidate.y : bestValue.y,
                           replace.z  ? candidate.z : bestValue.z,
                           replace.w  ? candidate.w : bestValue.w);
          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));
          srcIdx++;
        }
        setOutput(bestIndex);
      }
    `}}function Ba(n,e,t,s=null){let o=e.shape[0],r=e.shape[1];s!=null&&(o=s.shape[0],r=s.shape[1]);const a=jt(r),c={windowSize:a,inSize:r,batchSize:o,outSize:Math.ceil(r/a)},i=new am(c,t,s==null),l=[e];s!=null&&l.push(s);const u=n.runWebGLProgram(i,l,"int32");if(u.shape[1]===1)return u;const d=Ba(n,e,t,u);return n.disposeIntermediateTensorInfo(u),d}function Ma(n,e,t,s=null){const o=s!=null?s.shape:e.shape,r=o[o.length-1],a=jt(r),c=new im(o,a,t,s==null),i=s==null?[e]:[e,s],l=n.runWebGLProgram(c,i,"int32");if(l.shape.length===e.shape.length){const u=Ma(n,e,t,l);return n.disposeIntermediateTensorInfo(l),u}return l}function Wa(n,e,t,s){const o=[t];if(de("arg"+s.charAt(0).toUpperCase()+s.slice(1),o,e.shape.length),!w().getBool("WEBGL_PACK_REDUCE")||e.shape.length<=2){const r=[],a=n.texData.get(e.dataId),c=a!==null&&a.isPacked;let i=e;c&&(i=n.unpackTensor(e),r.push(i));const[l,u]=fe(i.shape,o),d=E(u),p=R({inputs:{x:i},backend:n,attrs:{shape:[-1,d]}});r.push(p);const h=Ba(n,p,s);r.push(h);const f=R({inputs:{x:h},backend:n,attrs:{shape:l}});return r.forEach(g=>n.disposeIntermediateTensorInfo(g)),f}return Ma(n,e,s)}function cm(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r}=s;let a=X(r,o.shape);const c=se(a,o.shape.length);let i=o;const l=[];c!=null&&(i=H({inputs:{x:o},backend:t,attrs:{perm:c}}),l.push(i),a=oe(a.length,i.shape.length)),de("argMax",[a[0]],i.shape.length);const u=Wa(t,i,a[0],"max");return l.forEach(d=>t.disposeIntermediateTensorInfo(d)),u}const lm={kernelName:ji,backendName:"webgl",kernelFunc:cm};function um(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r}=s;let a=X(r,o.shape);const c=se(a,o.shape.length);let i=o;const l=[];c!=null&&(i=H({inputs:{x:o},backend:t,attrs:{perm:c}}),l.push(i),a=oe(a.length,i.shape.length)),de("argMin",[a[0]],i.shape.length);const u=Wa(t,i,a[0],"min");return l.forEach(d=>t.disposeIntermediateTensorInfo(d)),u}const dm={kernelName:qi,backendName:"webgl",kernelFunc:um};const pm=ie+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return asin(x);
`,hm=D({opSnippet:pm}),fm={kernelName:Yi,backendName:"webgl",kernelFunc:hm};const mm=ie+"return log(x + sqrt(x * x + 1.0));",xm=D({opSnippet:mm}),gm={kernelName:Qi,backendName:"webgl",kernelFunc:xm};const Cm=ie+`
  return atan(x);
`,$m=D({opSnippet:Cm}),bm={kernelName:Zi,backendName:"webgl",kernelFunc:$m};const vm=hs+`
  return atan(a, b);
`,wm=`
  vec4 result = atan(a, b);
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Be+`
  return result;
`,Im=V({opSnippet:vm,packedOpSnippet:wm}),Rm={kernelName:Ji,backendName:"webgl",kernelFunc:Im};const ym=ie+`
  if ((x < -1.0) || (x > 1.0)) return NAN;
return (log(1.0 + x) - log(1.0 - x)) / 2.0;`,Sm=D({opSnippet:ym}),Tm={kernelName:ec,backendName:"webgl",kernelFunc:Sm};class gt{constructor(e,t,s,o=!1,r=!1){if(this.variableNames=["x"],t==="avg"&&s)throw new Error("Cannot compute positions for average pool.");const a=e.filterWidth,c=e.strideHeight,i=e.strideWidth,l=e.dilationHeight,u=e.dilationWidth,d=e.effectiveFilterHeight,p=e.effectiveFilterWidth,h=e.padInfo.top,f=e.padInfo.left;this.outputShape=e.outShape;const g=t==="avg",x=`((batch  * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + d`,m=`(xR * ${e.inWidth} + xC) * ${e.inChannels} + d`;let C="0.0";if(g||(C="-1.0 / 1e-20"),s){this.userCode=`
        const ivec2 strides = ivec2(${c}, ${i});
        const ivec2 pads = ivec2(${h}, ${f});

        void main() {
          ivec4 coords = getOutputCoords();
          int batch = coords[0];
          int d = coords[3];

          ivec2 xRCCorner = coords.yz * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${d};
              wR += ${l}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${p};
                wC += ${u}) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float value = getX(batch, xR, xC, d);

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value >= currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = ${o?r?x:m:`wR * ${p} + wC`};
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}const $="max";let b=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t==="avg"&&(b="avgValue / max(count, 1.0)");const v=Math.floor(a/4)*4,T=a%4,S=`
      if (${g}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${$}(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec2 strides = ivec2(${c}, ${i});
      const ivec2 pads = ivec2(${h}, ${f});
      const float initializationValue = ${C};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${C});
        float avgValue = 0.0;
        count = 0.0;

        for (int wR = 0; wR < ${d};
            wR += ${l}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${v}; wC += 4) {
            int xC = xCCorner + wC * ${u};

            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              getValue(batch, xR, xC + 2 * ${u}, d),
              getValue(batch, xR, xC + 3 * ${u}, d)
            );

            ${S}
          }

          int xC = xCCorner + ${v};
          if (${T===1}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );

            ${S}
          } else if (${T===2}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              initializationValue,
              initializationValue
            );

            ${S}
          } else if (${T===3}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              getValue(batch, xR, xC + 2 * ${u}, d),
              initializationValue
            );

            ${S}
          }
        }
        setOutput(${b});
      }
    `}}class ms{constructor(e,t,s,o=!1,r=!1){if(this.variableNames=["x"],t==="avg"&&s)throw new Error("Cannot compute positions for average pool.");const a=e.filterWidth,c=e.strideDepth,i=e.strideHeight,l=e.strideWidth,u=e.dilationDepth,d=e.dilationHeight,p=e.dilationWidth,h=e.effectiveFilterDepth,f=e.effectiveFilterHeight,g=e.effectiveFilterWidth,x=e.padInfo.front,m=e.padInfo.top,C=e.padInfo.left;this.outputShape=e.outShape;const $=t==="avg";let b="0.0";if($||(b="-1.0 / 1e-20"),s){this.userCode=`
        const ivec3 strides =
            ivec3(${c}, ${i}, ${l});
        const ivec3 pads = ivec3(${x}, ${m}, ${C});

        void main() {
          ivec5 coords = getOutputCoords();
          int batch = coords.x;
          int ch = coords.u;

          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
          int xDCorner = xCorner.x;
          int xRCorner = xCorner.y;
          int xCCorner = xCorner.z;

          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;

          for (int wD = 0; wD < ${h};
              wD += ${u}) {
            int xD = xDCorner + wD;

            if (xD < 0 || xD >= ${e.inDepth}) {
              continue;
            }

            for (int wR = 0; wR < ${f};
                wR += ${d}) {
              int xR = xRCorner + wR;

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int wC = 0; wC < ${g};
                  wC += ${p}) {
                int xC = xCCorner + wC;

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float value = getX(batch, xD, xR, xC, ch);

                // If a min / max value has already been found, use it. If not,
                // use the current value.
                float currMinMaxValue = mix(
                    value, minMaxValue, minMaxValueFound);
                if (value >= currMinMaxValue) {
                  minMaxValue = value;
                  minMaxValueFound = 1.0;
                  minMaxPosition = ${o?r?`(((batch * ${e.inDepth} + xD) * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`((xD * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`wD * ${f} * ${g} +
                      wR * ${g} + wC`};
                }
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}const v="max";let T=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t==="avg"&&(T="avgValue / max(count, 1.0)");const S=Math.floor(a/4)*4,I=a%4,A=`
      if (${$}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${v}(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec3 strides =
        ivec3(${c}, ${i}, ${l});
      const ivec3 pads = ivec3(${x}, ${m}, ${C});
      const float initializationValue = ${b};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xD, int xR, int xC, int ch) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xD, xR, xC, ch);
      }

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xDCorner = xCorner.x;
        int xRCorner = xCorner.y;
        int xCCorner = xCorner.z;

        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).
        // ? = to be determined
        vec4 minMaxValue = vec4(${b});
        float avgValue = 0.0;
        count = 0.0;

        for (int wD = 0; wD < ${h};
            wD += ${u}) {
          int xD = xDCorner + wD;

          if (xD < 0 || xD >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${f};
            wR += ${d}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${S}; wC += 4) {
              int xC = xCCorner + wC * ${p};

              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${p}, ch),
                getValue(batch, xD, xR, xC + 2 * ${p}, ch),
                getValue(batch, xD, xR, xC + 3 * ${p}, ch)
              );

              ${A}
            }

            int xC = xCCorner + ${S};
            if (${I===1}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                initializationValue,
                initializationValue,
                initializationValue
              );

              ${A}
            } else if (${I===2}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${p}, ch),
                initializationValue,
                initializationValue
              );

              ${A}
            } else if (${I===3}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${p}, ch),
                getValue(batch, xD, xR, xC + 2 * ${p}, ch),
                initializationValue
              );

              ${A}
            }
          }
        }
        setOutput(${T});
      }
    `}}function Em(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e;Ye(o,"avgPool");const{filterSize:r,strides:a,pad:c,dimRoundingMode:i}=s,l=1;O(Ke(a,l),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${a} and dilations '${l}'`);const u=Xe(o.shape,r,a,l,c,i);if(u.filterWidth===1&&u.filterHeight===1&&q(u.inShape,u.outShape))return Z({inputs:{x:o},backend:t});const d=new gt(u,"avg",!1);return t.runWebGLProgram(d,[o],"float32")}const Nm={kernelName:tc,backendName:"webgl",kernelFunc:Em};function km(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{filterSize:r,strides:a,pad:c,dimRoundingMode:i,dataFormat:l}=s,u=[1,1,1],d=$t(o.shape,r,a,u,c,i,l),p=new ms(d,"avg",!1);return t.runWebGLProgram(p,[o],"float32")}const Am={kernelName:nc,backendName:"webgl",kernelFunc:km};class Om{constructor(e){this.variableNames=["dy"],this.outputShape=e.inShape;const t=e.filterHeight,s=e.filterWidth,o=e.strideHeight,r=e.strideWidth,a=e.dilationHeight,c=e.dilationWidth,i=e.effectiveFilterHeight,l=e.effectiveFilterWidth,u=i-1-e.padInfo.top,d=l-1-e.padInfo.left,p=1/(t*s);this.userCode=`
      const ivec2 pads = ivec2(${u}, ${d});
      const float avgMultiplier = float(${p});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${i};
            wR += ${a}) {
          float dyR = float(dyRCorner + wR) / ${o}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${l};
            wC+= ${c}) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);

            dotProd += dyValue * avgMultiplier;
          }
        }
        setOutput(dotProd);
      }
    `}}class Dm{constructor(e){this.variableNames=["dy"],this.outputShape=e.inShape;const t=e.filterDepth,s=e.filterHeight,o=e.filterWidth,r=e.strideDepth,a=e.strideHeight,c=e.strideWidth,i=e.dilationDepth,l=e.dilationHeight,u=e.dilationWidth,d=e.effectiveFilterDepth,p=e.effectiveFilterHeight,h=e.effectiveFilterWidth,f=d-1-e.padInfo.front,g=p-1-e.padInfo.top,x=h-1-e.padInfo.left,m=1/(t*s*o);this.userCode=`
      const ivec3 pads = ivec3(${f}, ${g}, ${x});
      const float avgMultiplier = float(${m});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${d};
            wD += ${i}) {
          float dyD = float(dyDCorner + wD) / ${r}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${p};
              wR += ${l}) {
            float dyR = float(dyRCorner + wR) / ${a}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${h};
                wC += ${u}) {
              float dyC = float(dyCCorner + wC) / ${c}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);

              dotProd += dyValue * avgMultiplier;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}function Fm(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,input:r}=e,a=r,{filterSize:c,strides:i,pad:l,dimRoundingMode:u}=s,d=[1,1,1],p=$t(a.shape,c,i,d,l,u),h=new Dm(p);return t.runWebGLProgram(h,[o],a.dtype)}const Pm={kernelName:sc,backendName:"webgl",kernelFunc:Fm};function _m(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,input:r}=e,a=r;Ye([o,r],"avgPoolGrad");const{filterSize:c,strides:i,pad:l}=s,u=Xe(a.shape,c,i,1,l),d=new Om(u);return t.runWebGLProgram(d,[o],a.dtype)}const Lm={kernelName:oc,backendName:"webgl",kernelFunc:_m};function Vm(n){const{inputs:e,backend:t,attrs:s}=n,{a:o,b:r}=e,{transposeA:a,transposeB:c}=s;return Wt({a:o,b:r,transposeA:a,transposeB:c,backend:t})}const Bm={kernelName:rc,backendName:"webgl",kernelFunc:Vm};class Mm{constructor(e,t,s,o,r,a){this.outputShape=[],this.variableNames=["x","mean","variance"],z(e,t),z(e,s);let c="0.0";o!=null&&(z(e,o),this.variableNames.push("offset"),c="getOffsetAtOutCoords()");let i="1.0";r!=null&&(z(e,r),this.variableNames.push("scale"),i="getScaleAtOutCoords()"),this.outputShape=e,this.userCode=`
      void main() {
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${c};
        float scale = ${i};
        float inv = scale * inversesqrt(variance + float(${a}));
        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
    `}}class Wm{constructor(e,t,s,o,r,a){this.packedInputs=!0,this.packedOutput=!0,this.variableNames=["x","mean","variance"],z(e,t),z(e,s);let c="vec4(0.0)";o!=null&&(z(e,o),this.variableNames.push("offset"),c="getOffsetAtOutCoords()");let i="vec4(1.0)";r!=null&&(z(e,r),this.variableNames.push("scale"),i="getScaleAtOutCoords()"),this.outputShape=e,this.userCode=`
      void main() {
        vec4 offset = ${c};
        vec4 scale = ${i};

        vec4 x = getXAtOutCoords();
        vec4 mean = getMeanAtOutCoords();
        vec4 variance = getVarianceAtOutCoords();

        vec4 inv = scale * inversesqrt(variance + vec4(${a}));

        setOutput((x - mean) * inv + offset);
      }
    `}}const Um=({inputs:n,backend:e,attrs:t})=>{const{x:s,mean:o,variance:r,offset:a,scale:c}=n;O(o.shape.length===r.shape.length,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),O(a==null||o.shape.length===a.shape.length,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),O(c==null||o.shape.length===c.shape.length,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");let{varianceEpsilon:i}=t;i==null&&(i=.001);const l=[s,o,r];let u=null;a!=null&&(u=a.shape,l.push(a));let d=null;c!=null&&(d=c.shape,l.push(c));const p=w().getBool("WEBGL_PACK_NORMALIZATION")?new Wm(s.shape,o.shape,r.shape,u,d,i):new Mm(s.shape,o.shape,r.shape,u,d,i);return e.runWebGLProgram(p,l,l[0].dtype)},Gm={kernelName:ac,backendName:"webgl",kernelFunc:Um};class zm{constructor(e){this.variableNames=["source"],this.outputShape=e,this.rank=e.length;const t=_(this.rank);this.customUniforms=[{name:"start",arrayIndex:this.rank,type:"int"}];const s=Hm(this.rank);let o;const r=e.map((a,c)=>`sourceLoc.${xn[c]} = start[${c}] + coords.${xn[c]};`);o=`
        ${t} sourceLoc;
        ${t} coords = getOutputCoords();
        ${r.join(`
`)}
      `,this.userCode=`
      void main() {
        ${o}
        setOutput(getSource(${s}));
      }
    `}}const xn=["x","y","z","w","u","v"];function Hm(n){if(n===1)return"sourceLoc";if(n<=6)return xn.slice(0,n).map(e=>"sourceLoc."+e).join(",");throw Error(`Slicing for rank ${n} is not yet supported`)}class Xm{constructor(e){this.variableNames=["source"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.customUniforms=[{name:"start",arrayIndex:this.rank,type:"int"}];const t=_(this.rank),s=G("coords",this.rank),o=G("sourceLoc",this.rank),r=this.rank===1?"sourceLoc":`vec2(${o.slice(-2).join()})`,a=`getChannel(getSource(${o.join()}), ${r})`,c=`
      result.x = ${a};
      if (++${s[this.rank-1]} < ${e[this.rank-1]}) {
        ++${o[this.rank-1]};
        result.y = ${a};
        --${o[this.rank-1]};
      }
    `,i=this.rank===1?"":`
      --${s[this.rank-1]};
      if (++${s[this.rank-2]} < ${e[this.rank-2]}) {
        ++${o[this.rank-2]};
        result.z = ${a};
        if (++${s[this.rank-1]} < ${e[this.rank-1]}) {
          ++${o[this.rank-1]};
          result.w = ${a};
        }
      }
    `,l=this.rank<=4?`sourceLoc = coords +
            ${t}(${e.map((u,d)=>`start[${d}]`).join()});`:e.map((u,d)=>`${o[d]} = ${s[d]} + start[${d}];`).join(`
`);this.userCode=`
      void main() {
        ${t} coords = getOutputCoords();
        ${t} sourceLoc;
        ${l}
        vec4 result = vec4(0.);
        ${c}
        ${i}
        setOutput(result);
      }
    `}}function Km(n,e,t,s){const o=s.texData.get(n.dataId),r=s.makeTensorInfo(t,n.dtype),a=s.texData.get(r.dataId);Object.assign(a,o),a.refCount=1,a.shape=t,a.dtype=n.dtype;let c=jn(e,M(n.shape));o.slice&&(c+=o.slice.flatOffset),a.slice={flatOffset:c,origDataId:o.slice&&o.slice.origDataId||n.dataId};const i=s.dataRefCount.get(a.slice.origDataId)||1;return s.dataRefCount.set(a.slice.origDataId,i+1),r}function st(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{begin:r,size:a}=s,[c,i]=qn(o,r,a);if(Xn(o,c,i),E(i)===0)return t.makeTensorInfo(i,o.dtype,[]);if(t.shouldExecuteOnCPU([o])||o.dtype==="string"){const d=t.texData.get(o.dataId),p=_h(d.values,c,i,o.shape,o.dtype);return t.makeTensorInfo(i,o.dtype,p)}const{isPacked:l}=t.texData.get(o.dataId),u=Kn(o.shape,c,i);if(l||!u){const d=w().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new Xm(i):new zm(i),p=[c];return t.runWebGLProgram(d,[o],o.dtype,p)}return t.uploadToGPU(o.dataId),Km(o,c,i,t)}const jm={kernelName:lo,backendName:"webgl",kernelFunc:st};const qm=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{blockShape:r,crops:a}=s;O(o.shape.length<=4,()=>"batchToSpaceND for rank > 4 with a WebGL backend not implemented yet");const c=r.reduce((C,$)=>C*$),i=Qn(o.shape,r,c),l=Zn(i.length,r.length),u=Jn(o.shape,r,c),d=Eo(a,r.length),p=No(u,a,r.length),h=[],f=R({inputs:{x:o},backend:t,attrs:{shape:i}}),g=H({inputs:{x:f},backend:t,attrs:{perm:l}}),x=R({inputs:{x:g},backend:t,attrs:{shape:u}}),m=st({inputs:{x},backend:t,attrs:{begin:d,size:p}});return h.push(f),h.push(g),h.push(x),h.forEach(C=>t.disposeIntermediateTensorInfo(C)),m},Ym={kernelName:ic,backendName:"webgl",kernelFunc:qm};function Qm(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,weights:r}=e,{size:a}=s,c=t.readSync(o.dataId),i=t.readSync(r.dataId),l=Ta(c,i,r.dtype,r.shape,a);return t.makeTensorInfo([a],r.dtype,l)}const Zm={kernelName:cc,backendName:"webgl",kernelFunc:Qm};const Jm=`
  int r = int(a.r) & int(b.r);
  int g = int(a.g) & int(b.g);
  int rb = int(a.b) & int(b.b);
  int ra = int(a.a) & int(b.a);
  return vec4(r, g, rb, ra);
`,ex=`
  return float(int(a.r) & int(b.r));
`;function tx(n){const{inputs:e,backend:t}=n,{a:s,b:o}=e,r=w().getBool("WEBGL_PACK_BINARY_OPERATIONS"),a=w().getNumber("WEBGL_VERSION");if(t.shouldExecuteOnCPU([s,o])||a===1){const i=t.texData.get(s.dataId).values,l=t.texData.get(o.dataId).values,[u,d]=ah(s.shape,o.shape,i,l,s.dtype),p=t.makeTensorInfo(d,s.dtype),h=t.texData.get(p.dataId);return h.values=u,p}let c;return r?c=new tt(Jm,s.shape,o.shape,!1):c=new Fe(ex,s.shape,o.shape),t.runWebGLProgram(c,[s,o],s.dtype)}const nx={kernelName:vn,backendName:"webgl",kernelFunc:tx};function sx(n){const{inputs:e,backend:t}=n,{s0:s,s1:o}=e,r=t.readSync(s.dataId),a=t.readSync(o.dataId),c=z(Array.from(r),Array.from(a));return t.makeTensorInfo([c.length],"int32",Int32Array.from(c))}const ox={kernelName:lc,backendName:"webgl",kernelFunc:sx};const rx="return float(a != b);",Ua=V({opSnippet:rx,cpuKernelImpl:Th,dtype:"bool"}),ax={kernelName:_n,backendName:"webgl",kernelFunc:Ua};function yt(n){const{inputs:e,backend:t}=n,{input:s}=e,o=t.texData.get(s.dataId);return Z({inputs:{x:o.complexTensorInfos.real},backend:t})}const ix={kernelName:no,backendName:"webgl",kernelFunc:yt};const cx="return float(int(x));";function lx(n,e){const t=new he(n.shape,cx),s=e.runWebGLProgram(t,[n],"int32");return{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}function gn(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{dtype:r}=s;if(r==="complex64"){if(o.dtype==="complex64")return Z({inputs:{x:o},backend:t});const a=uc(o.shape),c=gn({inputs:{x:o},backend:t,attrs:{dtype:"float32"}}),i=Te({inputs:{real:c,imag:a},backend:t});return a.dispose(),t.disposeIntermediateTensorInfo(c),i}if(o.dtype==="complex64"){const a=yt({inputs:{input:o},backend:t}),c=gn({inputs:{x:a},backend:t,attrs:{dtype:r}});return t.disposeIntermediateTensorInfo(a),c}if(!oo(o.dtype,r)){const a=Z({inputs:{x:o},backend:t});return{dataId:a.dataId,shape:a.shape,dtype:r}}if(t.shouldExecuteOnCPU([o])){const a=t.texData.get(o.dataId).values,[c,i,l]=ih(a,o.shape,o.dtype,r);return t.makeTensorInfo(c,i,l)}if(r==="int32")return lx(o,t);if(r==="bool"){const a=t.makeTensorInfo([],"bool",ge("bool",1)),i=Ua({inputs:{a:o,b:a},backend:t});return t.disposeIntermediateTensorInfo(a),i}throw new Error(`Error in Cast: failed to cast ${o.dtype} to ${r}`)}const ux={kernelName:so,backendName:"webgl",kernelFunc:gn};const Ms="return ceil(x);",dx=D({opSnippet:Ms,packedOpSnippet:Ms,cpuKernelImpl:ch}),px={kernelName:wn,backendName:"webgl",kernelFunc:dx};class hx{constructor(e){this.variableNames=["A"],this.customUniforms=[{name:"minVal",type:"float"},{name:"maxVal",type:"float"}],this.outputShape=e,this.userCode=`

      void main() {
        float value = getAAtOutCoords();
        if (isnan(value)) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, minVal, maxVal));
      }
    `}}class fx{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"minVal",type:"float"},{name:"maxVal",type:"float"}],this.outputShape=e,this.userCode=`
      void main() {
        vec4 value = getAAtOutCoords();

        if (any(isnan(value))) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));
      }
    `}}function mx(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{clipValueMin:r,clipValueMax:a}=s;let c;w().getBool("WEBGL_PACK_CLIP")?c=new fx(o.shape):c=new hx(o.shape);const i=[[r],[a]];return t.runWebGLProgram(c,[o],o.dtype,i)}const xx={kernelName:dc,backendName:"webgl",kernelFunc:mx};class gx{constructor(e){this.variableNames=["real","imag"],this.outputShape=e,this.userCode=`
      void main() {
        float re = abs(getRealAtOutCoords());
        float im = abs(getImagAtOutCoords());
        float mx = max(re, im);

        // sadly the length function in glsl is not underflow-safe
        // (at least not on Intel GPUs). So the safe solution is
        // to ensure underflow-safety in all cases.
        setOutput(
          mx == 0.0 ? 0.0 : mx * length(vec2(1, min(re, im)/mx))
        );
      }
    `}}function Ws(n,e){return{dataId:e.dataId,dtype:e.dtype,shape:n.shape}}function Cx(n){const{inputs:e,backend:t}=n,{x:s}=e,o=t.texData.get(s.dataId),r=new gx(s.shape),a=[Ws(s,o.complexTensorInfos.real),Ws(s,o.complexTensorInfos.imag)];return t.runWebGLProgram(r,a,a[0].dtype)}const $x={kernelName:pc,backendName:"webgl",kernelFunc:Cx};class bx{constructor(e){this.outputShape=[],this.outputShape=Ae(e,1),this.variableNames=e.map((a,c)=>`T${c}`);const t=new Array(e.length-1);t[0]=e[0][1];for(let a=1;a<t.length;a++)t[a]=t[a-1]+e[a][1];const s=[`if (yC < ${t[0]}) setOutput(getT0(yR, yC));`];for(let a=1;a<t.length;a++){const c=t[a-1];s.push(`else if (yC < ${t[a]}) setOutput(getT${a}(yR, yC-${c}));`)}const o=t.length,r=t[t.length-1];s.push(`else setOutput(getT${o}(yR, yC-${r}));`),this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int yR = coords.x;
        int yC = coords.y;

        ${s.join(`
        `)}
      }
    `}}class vx{constructor(e,t){this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[],this.outputShape=Ae(e,t);const s=this.outputShape,o=s.length,r=_(o),a=G("coords",o),c=["x","y","z","w","u","v"].slice(0,o);this.variableNames=e.map((g,x)=>`T${x}`);const i=new Array(e.length-1);i[0]=e[0][t];for(let g=1;g<i.length;g++)i[g]=i[g-1]+e[g][t];const l=c[t],u=c.slice(-2),d=c.join();let p=`if (${l} < ${i[0]}) {
        return getChannel(
            getT0(${d}), vec2(${u.join()}));
        }`;for(let g=1;g<i.length;g++){const x=i[g-1];p+=`
        if (${l} < ${i[g]}  && ${l} >= ${i[g-1]}) {
          return getChannel(
            getT${g}(${Ot(c,l,x)}),
            vec2(${Ot(u,l,x)}));
        }`}const h=i.length,f=i[i.length-1];p+=`
        return getChannel(
          getT${h}(${Ot(c,l,f)}),
          vec2(${Ot(u,l,f)}));`,this.userCode=`
      float getValue(${c.map(g=>"int "+g)}) {
        ${p}
      }

      void main() {
        ${r} coords = getOutputCoords();
        vec4 result = vec4(getValue(${a}), 0., 0., 0.);

        ${a[o-1]} = ${a[o-1]} + 1;
        if (${a[o-1]} < ${s[o-1]}) {
          result.g = getValue(${a});
        }

        ${a[o-2]} = ${a[o-2]} + 1;
        if (${a[o-2]} < ${s[o-2]}) {
          result.a = getValue(${a});
        }

        ${a[o-1]} = ${a[o-1]} - 1;
        if (${a[o-2]} < ${s[o-2]} &&
            ${a[o-1]} < ${s[o-1]}) {
          result.b = getValue(${a});
        }
        setOutput(result);
      }
    `}}function Ot(n,e,t){const s=n.indexOf(e);return n.map((r,a)=>a===s?`${r} - ${t}`:r).join()}function Jt(n){const{inputs:e,backend:t}=n,{input:s}=e,o=t.texData.get(s.dataId);return Z({inputs:{x:o.complexTensorInfos.imag},backend:t})}const wx={kernelName:hc,backendName:"webgl",kernelFunc:Jt};function lt(n,e,t){const s=n[0].dtype;if(s==="complex64"){const h=n.map(C=>yt({inputs:{input:C},backend:t})),f=n.map(C=>Jt({inputs:{input:C},backend:t})),g=lt(h,e,t),x=lt(f,e,t),m=Te({inputs:{real:g,imag:x},backend:t});return h.forEach(C=>t.disposeIntermediateTensorInfo(C)),f.forEach(C=>t.disposeIntermediateTensorInfo(C)),t.disposeIntermediateTensorInfo(g),t.disposeIntermediateTensorInfo(x),m}let o=t.shouldExecuteOnCPU(n);if(s==="string"&&(o=!0),o){const h=n.map(b=>{const T=[-1,E(b.shape.slice(e))];return R({inputs:{x:b},backend:t,attrs:{shape:T}})}),f=h.map(b=>({vals:t.readSync(b.dataId),shape:b.shape})),g=Ae(h.map(b=>b.shape),1),x=h[0].shape[0]===1,m=lh(f,g,s,x),C=Ae(n.map(b=>b.shape),e),$=t.makeTensorInfo(C,s,m);return h.forEach(b=>t.disposeIntermediateTensorInfo(b)),$}const r=n.filter(h=>E(h.shape)>0),a=w().getBool("WEBGL_PACK_ARRAY_OPERATIONS")&&r[0].shape.length>1;if(r.length===1){const h=a?new he(n[0].shape,Ie):new Re(n[0].shape,Ie);return t.runWebGLProgram(h,n,s)}const c=w().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER");if(r.length>c){const h=[];for(let g=0;g<r.length;g+=c){const x=r.slice(g,g+c);h.push(lt(x,e,t))}const f=lt(h,e,t);for(const g of h)t.disposeIntermediateTensorInfo(g);return f}if(a){const h=new vx(r.map(f=>f.shape),e);return t.runWebGLProgram(h,r,s)}const{tensors2D:i,outShape:l}=Ix(r,e,t),u=new bx(i.map(h=>h.shape)),d=t.runWebGLProgram(u,i,s);i.forEach(h=>t.disposeIntermediateTensorInfo(h));const p=R({inputs:{x:d},attrs:{shape:l},backend:t});return t.disposeIntermediateTensorInfo(d),p}function Ix(n,e,t){const s=Ae(n.map(r=>r.shape),e);return{tensors2D:n.map(r=>R({inputs:{x:r},attrs:{shape:[-1,E(r.shape.slice(e))]},backend:t})),outShape:s}}function Ga(n){const{inputs:e,backend:t,attrs:s}=n,{axis:o}=s,r=X(o,e[0].shape)[0],a=e.map(l=>l.shape);wo(a,r);const c=Ae(e.map(l=>l.shape),r);if(E(c)===0)return t.makeTensorInfo(c,e[0].dtype,[]);const i=e.filter(l=>E(l.shape)>0);return i.length===1?Z({inputs:{x:i[0]},backend:t}):lt(i,r,t)}const Rx={kernelName:fc,backendName:"webgl",kernelFunc:Ga};class za{constructor(e,t=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.outputShape=e.outShape;const a=e.padInfo.top,c=e.padInfo.left,i=e.strideHeight,l=e.strideWidth,u=e.dilationHeight,d=e.dilationWidth,p=e.filterHeight,h=e.filterWidth,f=Math.floor(e.inChannels/4)*4,g=e.inChannels%4,x=e.dataFormat==="channelsLast",m=x?1:2,C=x?2:3,$=x?3:1;let b="",v="";s&&(o?b=`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:r?b=`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:b=`
          float activation(float x) {
            ${s}
          }
        `,v="result = activation(result);");const T=t?"result += getBiasAtOutCoords();":"";t&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${b}

      const ivec2 strides = ivec2(${i}, ${l});
      const ivec2 pads = ivec2(${a}, ${c});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d2 = coords[${$}];

        ivec2 xRCCorner =
            ivec2(coords[${m}], coords[${C}]) * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${p}; wR++) {
          int xR = xRCorner + wR * ${u};

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${h}; wC++) {
            int xC = xCCorner + wC * ${d};

            if (xC < 0 || xC >= ${e.inWidth}) {
              continue;
            }

            for (int d1 = 0; d1 < ${f}; d1 += 4) {
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );

              if (${x}) {
                vec4 xValues = vec4(
                  getX(batch, xR, xC, d1),
                  getX(batch, xR, xC, d1 + 1),
                  getX(batch, xR, xC, d1 + 2),
                  getX(batch, xR, xC, d1 + 3)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec4 xValues = vec4(
                  getX(batch, d1, xR, xC),
                  getX(batch, d1 + 1, xR, xC),
                  getX(batch, d1 + 2, xR, xC),
                  getX(batch, d1 + 3, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }
            }

            if (${g===1}) {

              if (${x}) {
                dotProd +=
                    getX(batch, xR, xC, ${f}) *
                    getW(wR, wC, ${f}, d2);
              } else {
                dotProd +=
                    getX(batch, ${f}, xR, xC) *
                    getW(wR, wC, ${f}, d2);
              }

            } else if (${g===2}) {
              vec2 wValues = vec2(
                getW(wR, wC, ${f}, d2),
                getW(wR, wC, ${f} + 1, d2)
              );

              if (${x}) {
                vec2 xValues = vec2(
                  getX(batch, xR, xC, ${f}),
                  getX(batch, xR, xC, ${f} + 1)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec2 xValues = vec2(
                  getX(batch, ${f}, xR, xC),
                  getX(batch, ${f} + 1, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            } else if (${g===3}) {
              vec3 wValues = vec3(
                getW(wR, wC, ${f}, d2),
                getW(wR, wC, ${f} + 1, d2),
                getW(wR, wC, ${f} + 2, d2)
              );

              if (${x}) {
                vec3 xValues = vec3(
                  getX(batch, xR, xC, ${f}),
                  getX(batch, xR, xC, ${f} + 1),
                  getX(batch, xR, xC, ${f} + 2)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec3 xValues = vec3(
                  getX(batch, ${f}, xR, xC),
                  getX(batch, ${f} + 1, xR, xC),
                  getX(batch, ${f} + 2, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            }
          }
        }

        float result = dotProd;
        ${T}
        ${v}
        setOutput(result);
      }
    `}}class yx{constructor(e){this.variableNames=["x","W"],this.outputShape=e.outShape;const t=e.padInfo.front,s=e.padInfo.top,o=e.padInfo.left,r=e.strideDepth,a=e.strideHeight,c=e.strideWidth,i=e.dilationDepth,l=e.dilationHeight,u=e.dilationWidth,d=e.filterDepth,p=e.filterHeight,h=e.filterWidth,f=Math.floor(e.inChannels/4)*4,g=e.inChannels%4;this.userCode=`
      const ivec3 strides = ivec3(${r}, ${a}, ${c});
      const ivec3 pads = ivec3(${t}, ${s}, ${o});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d2 = coords.u;

        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xFCorner = xFRCCorner.x;
        int xRCorner = xFRCCorner.y;
        int xCCorner = xFRCCorner.z;

        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get
        // y(yF, yR, yC, d2). ? = to be determined. : = across all
        // values in that axis.
        float dotProd = 0.0;
        for (int wF = 0; wF < ${d}; wF++) {
          int xF = xFCorner + wF * ${i};

          if (xF < 0 || xF >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${p}; wR++) {
            int xR = xRCorner + wR * ${l};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${h}; wC++) {
              int xC = xCCorner + wC * ${u};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              for (int d1 = 0; d1 < ${f}; d1 += 4) {
                vec4 xValues = vec4(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                vec4 wValues = vec4(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (${g===1}) {
                dotProd +=
                  getX(batch, xF, xR, xC, ${f}) *
                  getW(wF, wR, wC, ${f}, d2);
              } else if (${g===2}) {
                vec2 xValues = vec2(
                  getX(batch, xF, xR, xC, ${f}),
                  getX(batch, xF, xR, xC, ${f} + 1)
                );
                vec2 wValues = vec2(
                  getW(wF, wR, wC, ${f}, d2),
                  getW(wF, wR, wC, ${f} + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (${g===3}) {
                vec3 xValues = vec3(
                  getX(batch, xF, xR, xC, ${f}),
                  getX(batch, xF, xR, xC, ${f} + 1),
                  getX(batch, xF, xR, xC, ${f} + 2)
                );
                vec3 wValues = vec3(
                  getW(wF, wR, wC, ${f}, d2),
                  getW(wF, wR, wC, ${f} + 1, d2),
                  getW(wF, wR, wC, ${f} + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class Ha{constructor(e,t=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=e.outShape,this.enableShapeUniforms=W(this.outputShape.length);const a=e.padInfo.left,c=e.strideWidth,i=e.dilationWidth,l=e.filterHeight,u=e.filterWidth,d=u;let p=`
       int xR; int xC; int xCOffset;
       vec4 wTexel; vec4 previous; vec4 final;`;for(let x=0;x<u;x++)p+=`
           vec4 xTexelC${x*2};
           int xTexelC${x*2}Ready;
           vec4 xTexelC${x*2+1};
           int xTexelC${x*2+1}Ready;
           vec4 xC${x};`;p+=`
     for (int r = 0; r < ${l}; r++) {
      for (int d1 = 0; d1 < ${e.inChannels}; d1 += 2) {
       `;for(let x=0;x<u;x++)p+=`
           xTexelC${x*2} = vec4(0.0);
           xTexelC${x*2}Ready = 0;
           xTexelC${x*2+1} = vec4(0.0);
           xTexelC${x*2+1}Ready = 0;
           xC${x} = vec4(0.0);`;p+=`
         xR = xRCorner + r * dilations[0];
         if (xR >=0 && xR < inDims[0]) {
       `;for(let x=0;x<(d+1)/2;x++){const m=x*2;if(p+=`
           xC = xCCorner + ${m*i};
           `,c===1){if(m<u&&(a%2===1?(p+=`
                 xCOffset = xC + 1;
                 if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m}Ready == 0) {
                   xTexelC${m} = getX(batch, xR, xCOffset, d1);

                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${m}.zw = vec2(0.0);
                   }
                   xTexelC${m}Ready = 1;
                 }
               `,i===1&&m>0?p+=`
                 xC${m} = vec4(xTexelC${m-2}.zw, xTexelC${m}.xy);
                 `:p+=`
                   xCOffset = xC + 1 - 2;

                   if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       previous.zw = vec2(0.0);
                     }

                     xC${m} = vec4(previous.zw, xTexelC${m}.xy);
                   } else {
                     xC${m} = vec4(0.0, 0.0, xTexelC${m}.xy);
                   }
                   `):p+=`
                 if (xC >= 0 && xC < inDims[1] && xTexelC${m}Ready == 0) {
                   xTexelC${m} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${m}.zw = vec2(0.0);
                   }
                   xTexelC${m}Ready = 1;
                 }

                 xC${m} = xTexelC${m};
                 `,m+1<u)){const C=a%2===0?Gn(i):i;i%2===0&&a%2===1||i%2!==0&&a%2!==1?(p+=`
                   xCOffset = xC + imod(pads[1], 2) + ${C};

                   if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m+1}Ready == 0) {
                     xTexelC${m+1} = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       xTexelC${m+1}.zw = vec2(0.0);
                     }
                     xTexelC${m+1}Ready = 1;
                   }
                   `,i>1?p+=`
                     xCOffset -= 2;
                     if (xCOffset >= 0 && xCOffset < inDims[1]) {
                      previous = getX(batch, xR, xCOffset, d1);
                      xC${m+1} = vec4(previous.zw, xTexelC${m+1}.xy);
                     } else {
                      xC${m+1} = vec4(0.0, 0.0, xTexelC${m+1}.xy);
                     }
                     `:p+=`
                     xC${m+1} = vec4(xTexelC${m}.zw, xTexelC${m+1}.xy);
                     `):C===1?p+=`
                     xC${m+1} = xTexelC${m};
                     `:p+=`
                     xCOffset = xC + ${C};

                     if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m+1}Ready == 0) {
                       xTexelC${m+1} = getX(batch, xR, xCOffset, d1);
                       if (xCOffset + 1 >= inDims[1]) {
                         xTexelC${m+1}.zw = vec2(0.0);
                       }
                       xTexelC${m+1}Ready = 1;
                     }

                     xC${m+1} = xTexelC${m+1};
                     `}}else m<u&&(a%2===1?(p+=`
                 xCOffset = xC + 1 - strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m}Ready == 0) {
                   xTexelC${m} = getX(batch, xR, xCOffset, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${m}.zw = vec2(0.0);
                   }
                   xTexelC${m}Ready = 1;
                 }

                 if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${m+1}Ready == 0) {
                   xTexelC${m+1} = getX(batch, xR, xC + 1, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xC + 2 >= inDims[1]) {
                     xTexelC${m+1}.zw = vec2(0.0);
                   }
                   xTexelC${m+1}Ready = 1;
                 }

                 xC${m} = vec4(xTexelC${m}.zw, xTexelC${m+1}.zw);
               `,m+1<u&&(p+=`
                   final = vec4(0.0);
                   xCOffset = xC + 1 + strides[1];
                   if(xCOffset >= 0 && xCOffset < inDims[1]) {
                     final = getX(batch, xR, xCOffset, d1);
                   }
                   xC${m+1} = vec4(xTexelC${m+1}.xy, final.xy);
                 `)):(p+=`
                 if(xC >= 0 && xC < inDims[1] && xTexelC${m}Ready == 0) {
                   xTexelC${m} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${m}.zw = vec2(0.0);
                   }
                   xTexelC${m}Ready = 1;
                 }

                 xCOffset = xC + strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m+1}Ready == 0) {
                   xTexelC${m+1} = getX(batch, xR, xCOffset, d1);
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${m+1}.zw = vec2(0.);
                   }
                   xTexelC${m+1}Ready = 1;
                 }

                 xC${m} = vec4(
                   xTexelC${m}.xy, xTexelC${m+1}.xy);
               `,m+1<u&&(p+=`
                   xC${m+1} = vec4(xTexelC${m}.zw, xTexelC${m+1}.zw);
                 `)));m<u&&(p+=`
             wTexel = getW(r, ${m}, d1, d2);
             dotProd += xC${m}.xxzz * vec4(wTexel.xy, wTexel.xy);
             if(d1 + 1 < ${e.inChannels}) {
               dotProd += xC${m}.yyww * vec4(wTexel.zw, wTexel.zw);
             }
           `,m+1<u&&(p+=`
               wTexel = getW(r, ${m+1}, d1, d2);
               dotProd += xC${m+1}.xxzz * vec4(wTexel.xy, wTexel.xy);
               if(d1 + 1 < ${e.inChannels}) {
                 dotProd += xC${m+1}.yyww * vec4(wTexel.zw, wTexel.zw);
               }
             `))}p+=`
     }
   `,p+=`
     }
   `,p+=`
     }
   `;let h="",f="";s&&(o?h=`vec4 activation(vec4 a) {
           vec4 b = getPreluActivationWeightsAtOutCoords();
           ${s}
         }`:r?h=`vec4 activation(vec4 a) {
           vec4 b = getLeakyreluAlphaAtOutCoords();
           ${s}
         }`:h=`vec4 activation(vec4 x) {
           ${s}
         }`,f="result = activation(result);");const g=t?"result += getBiasAtOutCoords();":"";t&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
       ${h}

       void main() {
         ivec4 coords = getOutputCoords();
         int batch = coords.x;
         ivec2 xRCCorner = coords.yz * strides - pads;
         int d2 = coords.w;
         int xRCorner = xRCCorner.x;
         int xCCorner = xRCCorner.y;

         //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
         vec4 dotProd = vec4(0.000000000000001);

         ${p}

         vec4 result = dotProd - vec4(0.000000000000001);
         ${g}
         ${f}
         setOutput(result);
       }
     `}}class Sx{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"inputShape",type:"ivec4"},{name:"pad",type:"ivec2"},{name:"stride",type:"ivec2"},{name:"dilation",type:"ivec2"},{name:"inChannels",type:"int"},{name:"itemsPerBlockRow",type:"int"},{name:"outWidth",type:"int"}],this.outputShape=e,this.enableShapeUniforms=W(this.outputShape.length);const{dataFormat:s}=t,o=K(),r=s==="channelsLast",a=r?1:2,c=r?2:3,i=this.enableShapeUniforms?"if(blockIndex < outShape[2] && pos < outShape[1]) {":`if(blockIndex < ${e[2]} && pos < ${e[1]}) {`;let l="";for(let u=0;u<=1;u++)for(let d=0;d<=1;d++)l+=`
          blockIndex = rc.z + ${d};
          pos = rc.y + ${u};

          ${i}
            offsetY = int(blockIndex / outWidth) * stride[0] - pad[0];
            d0 = offsetY + dilation[0] * (pos / itemsPerBlockRow);

            if(d0 < inputShape[${a}] && d0 >= 0) {
              // Use custom imod instead mod. On Intel GPU, mod may generate
              // unexpected value.
              // https://github.com/tensorflow/tfjs/issues/5447
              offsetX = imod(blockIndex, outWidth) * stride[1] - pad[1];
              d1 = offsetX + dilation[1] * (imod(pos, itemsPerBlockRow) /
                  inChannels);

              if(d1 < inputShape[${c}] && d1 >= 0) {

                ch = imod(pos, inChannels);

                if (${r}) {
                  innerDims = vec2(d1, ch);
                  result[${u*2+d}] = getChannel(
                    getA(rc.x, d0, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                } else {
                  innerDims = vec2(d0, d1);
                  result[${u*2+d}] = getChannel(
                    getA(rc.x, ch, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                }
              }
            }
          }
        `;this.userCode=`
      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0);

        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
        vec2 innerDims;

        ${l}

        ${o.output} = result;
      }
    `}}function Ut(n,e){const t=n.length;return t>=3?e?[...n.slice(0,-3),n[t-3]*n[t-2],n[t-1]]:[...n.slice(0,-3),n[t-3],n[t-2]*n[t-1]]:!e&&t===1&&n[0]>1?[n[0],1]:null}function Xa({x:n,filter:e,convInfo:t,backend:s,bias:o=null,preluActivationWeights:r=null,leakyreluAlpha:a=0,activation:c=null}){const i=n.shape,l=s.texData.get(n.dataId),u=t.inChannels,d=i[0]*i[1]*i[2],p=t.outChannels,h=t.dataFormat==="channelsLast",f=!1,g=!1;let x;const m=[];if(r!=null){const b=Ut(r.shape,h);b!=null&&(r=R({inputs:{x:r},backend:s,attrs:{shape:b}}),m.push(r))}if(o!=null){const b=Ut(o.shape,h);b!=null&&(o=R({inputs:{x:o},backend:s,attrs:{shape:b}}),m.push(o))}if(!((d===1||p===1)&&u>Va)&&l.isPacked&&h&&l.texture!=null&&i[2]%2!==0&&q(l.shape.slice(-3),i.slice(-3))){const b=i[0]*i[1]*(i[2]+1),v={dataId:n.dataId,shape:[1,b,t.inChannels],dtype:n.dtype},T=l.shape;l.shape=l.shape.slice(),l.shape[l.shape.length-2]++,O(mt(l.shape,v.shape),()=>`packed reshape ${l.shape} to ${v.shape} isn't free`);const S=R({inputs:{x:e},backend:s,attrs:{shape:[1,t.inChannels,t.outChannels]}});m.push(S);const I=Wt({a:v,b:S,backend:s,transposeA:f,transposeB:g,bias:o,activation:c,preluActivationWeights:r,leakyreluAlpha:a}),A=s.texData.get(I.dataId);O(A.isPacked,()=>"batchMatMul result is expected to be packed"),l.shape=T,A.shape=t.outShape,x=Z({inputs:{x:I},backend:s}),x.shape=t.outShape,m.push(I)}else{const b=t.outHeight*t.outWidth,v=R({inputs:{x:n},backend:s,attrs:{shape:h?[t.batchSize,b,t.inChannels]:[t.batchSize,t.inChannels,b]}}),T=R({inputs:{x:e},backend:s,attrs:{shape:[1,t.inChannels,t.outChannels]}}),S=Wt({a:h?v:T,b:h?T:v,transposeA:!h,transposeB:g,backend:s,bias:o,activation:c,preluActivationWeights:r,leakyreluAlpha:a});x=R({inputs:{x:S},backend:s,attrs:{shape:t.outShape}}),m.push(v),m.push(T),m.push(S)}for(const b of m)s.disposeIntermediateTensorInfo(b);return x}function Ka({x:n,filter:e,convInfo:t,backend:s,bias:o=null,preluActivationWeights:r=null,leakyreluAlpha:a=0,activation:c=null}){const{filterWidth:i,filterHeight:l,inChannels:u,outWidth:d,outHeight:p,dataFormat:h}=t,f=h==="channelsLast",g=i*l*u,x=p*d,m=[t.batchSize,g,x],C=!0,$=!1,b=[];if(r!=null){const U=Ut(r.shape,f);U!=null&&(r=R({inputs:{x:r},backend:s,attrs:{shape:U}}),b.push(r))}if(o!=null){const U=Ut(o.shape,f);U!=null&&(o=R({inputs:{x:o},backend:s,attrs:{shape:U}}),b.push(o))}const v=R({inputs:{x:e},backend:s,attrs:{shape:[1,g,E(e.shape)/g]}});b.push(v);const T=new Sx(m,t),S=[n.shape,[t.padInfo.top,t.padInfo.left],[t.strideHeight,t.strideWidth],[t.dilationHeight,t.dilationWidth],[t.inChannels],[t.filterWidth*t.inChannels],[t.outWidth]],I=s.runWebGLProgram(T,[n],"float32",S),A=R({inputs:{x:I},backend:s,attrs:{shape:m}});b.push(I),b.push(A);const k=o!=null,F=r!=null,P=c==="leakyrelu",pe=c?xt(c,!0):null,Q=new La(f?A.shape:v.shape,f?v.shape:A.shape,f?[t.batchSize,x,t.outChannels]:[t.batchSize,t.outChannels,x],C,$,k,pe,F,P),ee=f?[A,v]:[v,A];if(o&&ee.push(o),F&&ee.push(r),P){const U=s.makeTensorInfo([],"float32",je(a,"float32"));ee.push(U),b.push(U)}const re=s.runWebGLProgram(Q,ee,"float32"),ce=R({inputs:{x:re},backend:s,attrs:{shape:t.outShape}});b.push(re);for(const U of b)s.disposeIntermediateTensorInfo(U);return ce}function Tx(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,filter:r}=e,{strides:a,pad:c,dataFormat:i,dilations:l,dimRoundingMode:u}=s,d=bt(i),p=$e(o.shape,r.shape,a,l,c,u,!1,d);let h;if(p.filterHeight===1&&p.filterWidth===1&&p.dilationHeight===1&&p.dilationWidth===1&&p.strideHeight===1&&p.strideWidth===1&&(p.padInfo.type==="SAME"||p.padInfo.type==="VALID"))h=Xa({x:o,filter:r,convInfo:p,backend:t});else if(p.strideWidth<=2&&d==="channelsLast"&&w().getBool("WEBGL_EXP_CONV")){const g=new Ha(p),x=[[p.padInfo.top,p.padInfo.left],[p.strideHeight,p.strideWidth],[p.dilationHeight,p.dilationWidth],[p.inHeight,p.inWidth]];h=t.runWebGLProgram(g,[o,r],"float32",x)}else if(w().getBool("WEBGL_CONV_IM2COL"))h=Ka({x:o,filter:r,convInfo:p,backend:t});else{const g=new za(p);h=t.runWebGLProgram(g,[o,r],"float32")}const f=R({inputs:{x:h},backend:t,attrs:{shape:p.outShape}});return t.disposeIntermediateTensorInfo(h),f}const Ex={kernelName:mc,backendName:"webgl",kernelFunc:Tx};class Nx{constructor(e){this.variableNames=["x","dy"],this.outputShape=e.filterShape;const t=e.strideHeight,s=e.strideWidth,o=e.padInfo.top,r=e.padInfo.left,a=e.dataFormat==="channelsLast";this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int d2 = coords.w;

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${o};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${s} - ${r};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              ${a?`float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);`:`float dyValue = getDy(b, d2, yR, yC);
              float xValue = getX(b, d1, xR, xC);
              dotProd += (xValue * dyValue);`}
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class kx{constructor(e){this.variableNames=["dy","W"],this.outputShape=e.inShape;const t=e.filterHeight,s=e.filterWidth,o=e.strideHeight,r=e.strideWidth,a=e.dataFormat==="channelsLast",c=t-1-e.padInfo.top,i=s-1-e.padInfo.left,l=a?1:2,u=a?2:3,d=a?3:1;this.userCode=`
      const ivec2 pads = ivec2(${c}, ${i});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[${d}];

        ivec2 dyCorner = ivec2(coords[${l}], coords[${u}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${o}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${s} - 1 - wC;

            for (int d2 = 0; d2 < ${e.outChannels}; d2++) {

              if (${a}) {
                float xValue = getDy(batch, idyR, idyC, d2);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              } else {
                float xValue = getDy(batch, d2, idyR, idyC);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }

            }
          }
        }
        setOutput(dotProd);
      }
    `}}class Ax{constructor(e){this.variableNames=["x","dy"],this.outputShape=e.filterShape;const t=e.strideDepth,s=e.strideHeight,o=e.strideWidth,r=e.padInfo.front,a=e.padInfo.top,c=e.padInfo.left;this.userCode=`
      void main() {
        ivec5 coords = getOutputCoords();
        int wF = coords.x;
        int wR = coords.y;
        int wC = coords.z;
        int d1 = coords.w;
        int d2 = coords.u;

        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yF = 0; yF < ${e.outDepth}; yF++) {
            int xF = wF + yF * ${t} - ${r};

            if (xF < 0 || xF >= ${e.inDepth}) {
              continue;
            }

            for (int yR = 0; yR < ${e.outHeight}; yR++) {
              int xR = wR + yR * ${s} - ${a};

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int yC = 0; yC < ${e.outWidth}; yC++) {
                int xC = wC + yC * ${o} - ${c};

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float dyValue = getDy(b, yF, yR, yC, d2);
                float xValue = getX(b, xF, xR, xC, d1);
                dotProd += (xValue * dyValue);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class Ox{constructor(e){this.variableNames=["dy","W"],this.outputShape=e.inShape;const t=e.filterDepth,s=e.filterHeight,o=e.filterWidth,r=e.strideDepth,a=e.strideHeight,c=e.strideWidth,i=t-1-e.padInfo.front,l=s-1-e.padInfo.top,u=o-1-e.padInfo.left;this.userCode=`
      const ivec3 pads = ivec3(${i}, ${l}, ${u});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.u;


        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyFCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        float dotProd = 0.0;
        for (int wF = 0; wF < ${t}; wF++) {
          float dyF = float(dyFCorner + wF) / ${r}.0;

          if (dyF < 0.0 || dyF >= ${e.outDepth}.0 || fract(dyF) > 0.0) {
            continue;
          }
          int idyF = int(dyF);

          int wFPerm = ${t} - 1 - wF;

          for (int wR = 0; wR < ${s}; wR++) {
            float dyR = float(dyRCorner + wR) / ${a}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
              fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            int wRPerm = ${s} - 1 - wR;

            for (int wC = 0; wC < ${o}; wC++) {
              float dyC = float(dyCCorner + wC) / ${c}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              int wCPerm = ${o} - 1 - wC;

              for (int d2 = 0; d2 < ${e.outChannels}; d2++) {
                float xValue = getDy(batch, idyF, idyR, idyC, d2);
                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}function Dx(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,dy:r}=e,{strides:a,pad:c,dataFormat:i,dimRoundingMode:l,filterShape:u}=s,d=bt(i),p=$e(o.shape,u,a,1,c,l,!1,d),h=new Nx(p);return t.runWebGLProgram(h,[o,r],"float32")}const Fx={kernelName:xc,backendName:"webgl",kernelFunc:Dx};class Px{constructor(e){this.variableNames=["dy","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"strides",type:"vec2"}],this.outputShape=e.inShape,this.enableShapeUniforms=W(this.outputShape.length);const t=e.filterHeight,s=e.filterWidth,o=t-1-e.padInfo.top,r=s-1-e.padInfo.left;this.userCode=`
      const ivec2 pads = ivec2(${o}, ${r});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];

        ivec2 dyCorner = ivec2(coords[1], coords[2]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        vec4 result = vec4(0.);
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / strides[0];
          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);
          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            int wCPerm = ${s} - 1 - wC;

            float dyC = float(dyCCorner + wC) / strides[1];
            bool idyCVal = (dyC >= 0.0) && (dyC < ${e.outWidth}.0)
              && (fract(dyC) == 0.0);
            int idyC = int(dyC);

            float dyC2 = float(dyCCorner + wC + 1) / strides[1];
            bool idyCVal2 = (dyC2 >= 0.0) && (dyC2 < ${e.outWidth}.0)
              && (fract(dyC2) == 0.0);
            int idyC2 = int(dyC2);

            if (idyCVal && idyCVal2) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec4 dySample2 = (idyC / 2 == idyC2 / 2) ?
                  dySample : getDy(batch, idyR, idyC2, d2);

                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));

                dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample2.xy : dySample2.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal2) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC2, d2);
                vec2 dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            }
          }
        }
        setOutput(result);
      }
    `}}function _x(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,filter:r}=e,{inputShape:a,strides:c,pad:i,dataFormat:l,dimRoundingMode:u}=s,d=bt(l),p=$e(a,r.shape,c,1,i,u,!1,d);if(w().getBool("WEBGL_PACK_CONV2DTRANSPOSE")&&d==="channelsLast"){const h=[[p.strideHeight,p.strideWidth]],f=new Px(p);return t.runWebGLProgram(f,[o,r],"float32",h)}else{const h=new kx(p);return t.runWebGLProgram(h,[o,r],"float32")}}const Lx={kernelName:gc,backendName:"webgl",kernelFunc:_x};function Vx(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,filter:r}=e,{strides:a,pad:c,dilations:i}=s,l=Xt(o.shape,r.shape,a,i,c),u=new yx(l);return t.runWebGLProgram(u,[o,r],"float32")}const Bx={kernelName:Cc,backendName:"webgl",kernelFunc:Vx};function Mx(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,dy:r}=e,{strides:a,pad:c,filterShape:i}=s,l=Xt(o.shape,i,a,1,c),u=new Ax(l);return t.runWebGLProgram(u,[o,r],"float32")}const Wx={kernelName:$c,backendName:"webgl",kernelFunc:Mx};function Ux(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,filter:r}=e,{pad:a,strides:c,inputShape:i}=s,l=Xt(i,r.shape,c,1,a),u=new Ox(l);return t.runWebGLProgram(u,[o,r],"float32")}const Gx={kernelName:bc,backendName:"webgl",kernelFunc:Ux};const zx=nt+`
  return cos(x);
`,Hx=`
  vec4 result = cos(x);
  bvec4 isNaN = isnan(x);
  ${Be}
  return result;
`,Xx=D({opSnippet:zx,packedOpSnippet:Hx}),Kx={kernelName:vc,backendName:"webgl",kernelFunc:Xx};const jx=`
  float e2x = exp(-x);
  return (e2x + 1.0 / e2x) / 2.0;
`,qx=D({opSnippet:jx}),Yx={kernelName:wc,backendName:"webgl",kernelFunc:qx};class Qx{constructor(e,t,s,o,r){this.variableNames=["Image","Boxes","BoxInd"],this.outputShape=[];const[a,c,i,l]=e,[u]=t,[d,p]=s;this.outputShape=[u,d,p,l];const h=o==="bilinear"?1:0,[f,g]=[`${c-1}.0`,`${i-1}.0`],[x,m,C]=d>1?[`${(c-1)/(d-1)}`,"(y2-y1) * height_ratio",`y1*${f} + float(y)*(height_scale)`]:["0.0","0.0",`0.5 * (y1+y2) * ${f}`],[$,b,v]=p>1?[`${(i-1)/(p-1)}`,"(x2-x1) * width_ratio",`x1*${g} + float(x)*(width_scale)`]:["0.0","0.0",`0.5 * (x1+x2) * ${g}`];this.userCode=`
      const float height_ratio = float(${x});
      const float width_ratio = float(${$});
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int y = coords[1];
        int x = coords[2];
        int d = coords[3];

        // get box vals
        float y1 = getBoxes(b,0);
        float x1 = getBoxes(b,1);
        float y2 = getBoxes(b,2);
        float x2 = getBoxes(b,3);

        // get image in batch index
        int bInd = round(getBoxInd(b));
        if(bInd < 0 || bInd >= ${a}) {
          return;
        }

        float height_scale = ${m};
        float width_scale = ${b};

        float in_y = ${C};
        if( in_y < 0.0 || in_y > ${f} ) {
          setOutput(float(${r}));
          return;
        }
        float in_x = ${v};
        if( in_x < 0.0 || in_x > ${g} ) {
          setOutput(float(${r}));
          return;
        }

        vec2 sourceFracIndexCR = vec2(in_x,in_y);
        if(${h} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);
          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));

          float topLeft = getImage(b, sourceFloorCR.y, sourceFloorCR.x, d);
          float bottomLeft = getImage(b, sourceCeilCR.y, sourceFloorCR.x, d);
          float topRight = getImage(b, sourceFloorCR.y, sourceCeilCR.x, d);
          float bottomRight = getImage(b, sourceCeilCR.y, sourceCeilCR.x, d);

          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);

          float top = topLeft + (topRight - topLeft) * fracCR.x;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          float newValue = top + (bottom - top) * fracCR.y;
          setOutput(newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestCR = ivec2(floor(
            sourceFracIndexCR + vec2(0.5,0.5)));
          float newValue = getImage(b, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutput(newValue);
        }
      }
    `}}const Zx=n=>{const{inputs:e,backend:t,attrs:s}=n,{image:o,boxes:r,boxInd:a}=e,{cropSize:c,method:i,extrapolationValue:l}=s,u=new Qx(o.shape,r.shape,c,i,l);return t.runWebGLProgram(u,[o,r,a],"float32")},Jx={kernelName:Ic,backendName:"webgl",kernelFunc:Zx};var Ct;(function(n){n.Prod="*",n.Sum="+"})(Ct||(Ct={}));class Us{constructor(e,t,s,o){this.op=e,this.outputShape=t,this.variableNames=["x"],this.customUniforms=[{name:"index",type:"float"}];const r=this.outputShape.length,a=this.op===Ct.Prod?"1.0":"0.0",c=s?a:`getX(${Gs(r,"coords",this.op)})`,i=this.outputShape[this.outputShape.length-1];let l="",u="";s?(l=o?`end != ${i-1}`:"end != 0",u=o?"end + 1":"end - 1"):(l=o?`end + pow2 < ${i}`:"end >= pow2",u=o?"end + pow2":"end - pow2"),this.userCode=`
      void main() {
        ${_(r)} coords = getOutputCoords();
        int end = ${zs(r,"coords",this.op)};
        float val = ${c};
        int pow2 = int(pow(2.0, index));
        if (${l}) {
          int idx = ${u};
          ${zs(r,"coords",this.op)} = idx;
          val ${this.op}= getX(${Gs(r,"coords",this.op)});
        }
        setOutput(val);
      }
    `}}function Gs(n,e,t){if(n===1)return`${e}`;if(n===2)return`${e}.x, ${e}.y`;if(n===3)return`${e}.x, ${e}.y, ${e}.z`;if(n===4)return`${e}.x, ${e}.y, ${e}.z, ${e}.w`;throw new Error(`Cumulative ${t} for rank ${n} is not yet supported`)}function zs(n,e,t){if(n===1)return`${e}`;if(n===2)return`${e}.y`;if(n===3)return`${e}.z`;if(n===4)return`${e}.w`;throw new Error(`Cumulative ${t} for rank ${n} is not yet supported`)}function ja(n,e,t,s,o,r){const a=e.shape.length,c=se([s],a);let i=e;c!=null&&(i=H({inputs:{x:e},backend:t,attrs:{perm:c}}));const l=oe(1,a)[0];if(l!==a-1)throw new Error(`WebGL cumprod shader expects an inner-most axis=${e.shape.length-1} but got axis=${s}`);const u=i.shape[l];let d=Z({inputs:{x:i},backend:t});for(let p=0;p<=Math.ceil(Math.log2(u))-1;p++){const h=new Us(n,i.shape,!1,r),f=[[p]],g=d;d=t.runWebGLProgram(h,[d],d.dtype,f),t.disposeIntermediateTensorInfo(g)}if(o){const p=new Us(n,i.shape,o,r),h=d;d=t.runWebGLProgram(p,[d],d.dtype),t.disposeIntermediateTensorInfo(h)}if(c!=null){const p=$n(c),h=H({inputs:{x:d},backend:t,attrs:{perm:p}});return t.disposeIntermediateTensorInfo(d),t.disposeIntermediateTensorInfo(i),h}return d}function eg(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,exclusive:a,reverse:c}=s;return ja(Ct.Prod,o,t,r,a,c)}const tg={kernelName:Rc,backendName:"webgl",kernelFunc:eg};function ng(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,exclusive:a,reverse:c}=s;return ja(Ct.Sum,o,t,r,a,c)}const sg={kernelName:yc,backendName:"webgl",kernelFunc:ng};function og(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,weights:r}=e,{size:a,binaryOutput:c}=s;if(o.shape.length===1){const i=t.readSync(o.dataId),l=t.readSync(r.dataId),u=Ta(i,l,r.dtype,r.shape,a);return t.makeTensorInfo([a],r.dtype,u)}else if(o.shape.length===2){const i=t.bufferSync(o),l=t.bufferSync(r),u=rh(i,l,a,c);return t.makeTensorInfo(u.shape,r.dtype,u.values)}throw new Error(`Error in denseBincount: input must be at most rank 2, but got rank${o.shape.length}.`)}const rg={kernelName:Sc,backendName:"webgl",kernelFunc:og};class ag{constructor(e,t,s){this.variableNames=["x"],this.outputShape=[],this.outputShape=e,this.blockSize=t,this.dataFormat=s,this.userCode=`
    void main() {
      ivec4 coords = getOutputCoords();
      int b = coords[0];
      int h = ${this.getHeightCoordString()};
      int w = ${this.getWidthCoordString()};
      int d = ${this.getDepthCoordString()};

      int in_h = h / ${t};
      int offset_h = imod(h, ${t});
      int in_w = w / ${t};
      int offset_w = imod(w, ${t});
      int offset_d = (offset_h * ${t} + offset_w) *
        ${this.getOutputDepthSize()};
      int in_d = d + offset_d;

      float result = ${this.getInputSamplingString()};
      setOutput(result);
    }
  `}getHeightCoordString(){return this.dataFormat==="NHWC"?"coords[1]":"coords[2]"}getWidthCoordString(){return this.dataFormat==="NHWC"?"coords[2]":"coords[3]"}getDepthCoordString(){return this.dataFormat==="NHWC"?"coords[3]":"coords[1]"}getOutputDepthSize(){return this.dataFormat==="NHWC"?this.outputShape[3]:this.outputShape[1]}getInputSamplingString(){return this.dataFormat==="NHWC"?"getX(b, in_h, in_w, in_d)":"getX(b, in_d, in_h, in_w)"}}function ig(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{blockSize:r,dataFormat:a}=s,c=o.shape[0],i=a==="NHWC"?o.shape[1]:o.shape[2],l=a==="NHWC"?o.shape[2]:o.shape[3],u=a==="NHWC"?o.shape[3]:o.shape[1],d=i*r,p=l*r,h=u/(r*r),f=a==="NHWC"?[c,d,p,h]:[c,h,d,p],g=new ag(f,r,a);return t.runWebGLProgram(g,[o],o.dtype)}const cg={kernelName:Tc,backendName:"webgl",kernelFunc:ig};class qa{constructor(e,t=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=e.outShape,this.enableShapeUniforms=W(this.outputShape.length);const a=e.filterHeight,c=e.filterWidth,i=e.outChannels/e.inChannels;let l="",u="";s&&(o?l=`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:r?l=`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:l=`
          float activation(float x) {
            ${s}
          }
        `,u="result = activation(result);");const d=t?"result += getBiasAtOutCoords();":"";t&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${l}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${i};
        int q = d2 - d1 * ${i};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < ${a}; wR++) {
          int xR = xRCorner + wR * dilations[0];

          if (xR < 0 || xR >= inDims[0]) {
            continue;
          }

          for (int wC = 0; wC < ${c}; wC++) {
            int xC = xCCorner + wC * dilations[1];

            if (xC < 0 || xC >= inDims[1]) {
              continue;
            }

            float xVal = getX(batch, xR, xC, d1);
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }

        float result = dotProd;
        ${d}
        ${u}
        setOutput(result);
      }
    `}}class Ya{constructor(e,t=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=e.outShape,this.enableShapeUniforms=W(this.outputShape.length);const a=e.outChannels/e.inChannels,c=e.padInfo.left,i=e.strideWidth,l=e.dilationWidth,u=e.filterHeight,d=e.filterWidth,p=d;let h=`
      int xR; int xC; int xCOffset;
      vec4 wTexel; vec4 previous; vec4 final;`;for(let m=0;m<d;m++)h+=`
          vec4 xTexelC${m*2};
          int xTexelC${m*2}Ready;
          vec4 xTexelC${m*2+1};
          int xTexelC${m*2+1}Ready;
          vec4 xC${m};`;h+=`
    for (int r = 0; r < ${u}; r++) {
      `;for(let m=0;m<d;m++)h+=`
          xTexelC${m*2} = vec4(0.0);
          xTexelC${m*2}Ready = 0;
          xTexelC${m*2+1} = vec4(0.0);
          xTexelC${m*2+1}Ready = 0;
          xC${m} = vec4(0.0);`;h+=`
        xR = xRCorner + r * dilations[0];
        if (xR >=0 && xR < inDims[0]) {
      `;for(let m=0;m<(p+1)/2;m++){const C=m*2;if(h+=`
          xC = xCCorner + ${C*l};
          `,i===1){if(C<d&&(c%2===1?(h+=`
                xCOffset = xC + 1;
                if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${C}Ready == 0) {
                  xTexelC${C} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${C}.zw = vec2(0.0);
                  }
                  xTexelC${C}Ready = 1;
                }
              `,l===1&&C>0?h+=`
                xC${C} = vec4(xTexelC${C-2}.zw, xTexelC${C}.xy);
                `:h+=`
                  xCOffset = xC + 1 - 2;

                  if (xCOffset >= 0 && xCOffset < inDims[1]) {
                    previous = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      previous.zw = vec2(0.0);
                    }

                    xC${C} = vec4(previous.zw, xTexelC${C}.xy);
                  } else {
                    xC${C} = vec4(0.0, 0.0, xTexelC${C}.xy);
                  }
                  `):h+=`
                if (xC >= 0 && xC < inDims[1] && xTexelC${C}Ready == 0) {
                  xTexelC${C} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${C}.zw = vec2(0.0);
                  }
                  xTexelC${C}Ready = 1;
                }

                xC${C} = xTexelC${C};
                `,C+1<d)){const $=c%2===0?Gn(l):l;l%2===0&&c%2===1||l%2!==0&&c%2!==1?(h+=`
                  xCOffset = xC + imod(pads[1], 2) + ${$};

                  if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${C+1}Ready == 0) {
                    xTexelC${C+1} = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      xTexelC${C+1}.zw = vec2(0.0);
                    }
                    xTexelC${C+1}Ready = 1;
                  }
                  `,l>1?h+=`
                    xCOffset -= 2;
                    if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);
                     xC${C+1} = vec4(previous.zw, xTexelC${C+1}.xy);
                    } else {
                     xC${C+1} = vec4(0.0, 0.0, xTexelC${C+1}.xy);
                    }
                    `:h+=`
                    xC${C+1} = vec4(xTexelC${C}.zw, xTexelC${C+1}.xy);
                    `):$===1?h+=`
                    xC${C+1} = xTexelC${C};
                    `:h+=`
                    xCOffset = xC + ${$};

                    if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${C+1}Ready == 0) {
                      xTexelC${C+1} = getX(batch, xR, xCOffset, d1);
                      if (xCOffset + 1 >= inDims[1]) {
                        xTexelC${C+1}.zw = vec2(0.0);
                      }
                      xTexelC${C+1}Ready = 1;
                    }

                    xC${C+1} = xTexelC${C+1};
                    `}}else C<d&&(c%2===1?(h+=`
                xCOffset = xC + 1 - strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${C}Ready == 0) {
                  xTexelC${C} = getX(batch, xR, xCOffset, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${C}.zw = vec2(0.0);
                  }
                  xTexelC${C}Ready = 1;
                }

                if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${C+1}Ready == 0) {
                  xTexelC${C+1} = getX(batch, xR, xC + 1, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xC + 2 >= inDims[1]) {
                    xTexelC${C+1}.zw = vec2(0.0);
                  }
                  xTexelC${C+1}Ready = 1;
                }

                xC${C} = vec4(xTexelC${C}.zw, xTexelC${C+1}.zw);
              `,C+1<d&&(h+=`
                  final = vec4(0.0);
                  xCOffset = xC + 1 + strides[1];
                  if(xCOffset >= 0 && xCOffset < inDims[1]) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xC${C+1} = vec4(xTexelC${C+1}.xy, final.xy);
                `)):(h+=`
                if(xC >= 0 && xC < inDims[1] && xTexelC${C}Ready == 0) {
                  xTexelC${C} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${C}.zw = vec2(0.0);
                  }
                  xTexelC${C}Ready = 1;
                }

                xCOffset = xC + strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${C+1}Ready == 0) {
                  xTexelC${C+1} = getX(batch, xR, xCOffset, d1);
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${C+1}.zw = vec2(0.);
                  }
                  xTexelC${C+1}Ready = 1;
                }

                xC${C} = vec4(
                  xTexelC${C}.xy, xTexelC${C+1}.xy);
              `,C+1<d&&(h+=`
                  xC${C+1} = vec4(xTexelC${C}.zw, xTexelC${C+1}.zw);
                `)));C<d&&(h+=`
            wTexel = getW(r, ${C}, d1, q);
            dotProd += xC${C} * vec4(wTexel.xz, wTexel.xz);
          `,C+1<d&&(h+=`
              wTexel = getW(r, ${C+1}, d1, q);
              dotProd += xC${C+1} * vec4(wTexel.xz, wTexel.xz);
            `))}h+=`
    }
  `,h+=`
      }
    `;let f="",g="";s&&(o?f=`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:r?f=`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:f=`vec4 activation(vec4 x) {
          ${s}
        }`,g="result = activation(result);");const x=t?"result += getBiasAtOutCoords();":"";t&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${f}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${a};
        int q = d2 - d1 * ${a};
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
        vec4 dotProd = vec4(0.000000000000001);

        ${h}

        vec4 result = dotProd - vec4(0.000000000000001);
        ${x}
        ${g}
        setOutput(result);
      }
    `}}function lg(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,filter:r}=e,{strides:a,pad:c,dilations:i,dimRoundingMode:l}=s;let u=i;u==null&&(u=[1,1]),O(Ke(a,u),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${a} and dilations '${u}'`);const d=$e(o.shape,r.shape,a,u,c,l,!0);let p;w().getBool("WEBGL_PACK_DEPTHWISECONV")&&d.strideWidth<=2&&d.outChannels/d.inChannels===1?p=new Ya(d):p=new qa(d);const h=[[d.padInfo.top,d.padInfo.left],[d.strideHeight,d.strideWidth],[d.dilationHeight,d.dilationWidth],[d.inHeight,d.inWidth]];return t.runWebGLProgram(p,[o,r],"float32",h)}const ug={kernelName:Ec,backendName:"webgl",kernelFunc:lg};class dg{constructor(e){this.variableNames=["x","dy"],this.outputShape=e.filterShape;const t=e.strideHeight,s=e.strideWidth,o=e.padInfo.top,r=e.padInfo.left,a=e.outChannels/e.inChannels;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int dm = coords.w;
        int d2 = d1 * ${a} + dm;

        float dotProd = 0.0;

        // TO DO: Vec4 over the batch size
        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${o};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${s} - ${r};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class pg{constructor(e){this.variableNames=["dy","W"],this.outputShape=e.inShape;const t=e.filterHeight,s=e.filterWidth,o=e.strideHeight,r=e.strideWidth,a=t-1-e.padInfo.top,c=s-1-e.padInfo.left,i=e.outChannels/e.inChannels;this.userCode=`
      const ivec2 pads = ivec2(${a}, ${c});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];
        ivec2 dyCorner = coords.yz - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        float dotProd = 0.0;

        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${o}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${s} - 1 - wC;

            // TO DO: Vec4 over the channelMul
            for (int dm = 0; dm < ${i}; dm++) {
              int d2 = d1 * ${i} + dm;
              float xValue = getDy(batch, idyR, idyC, d2);
              float wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}function hg(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,dy:r}=e,{strides:a,dilations:c,pad:i,dimRoundingMode:l,filterShape:u}=s,d=$e(o.shape,u,a,c,i,l,!0),p=new dg(d);return t.runWebGLProgram(p,[o,r],"float32")}const fg={kernelName:Nc,backendName:"webgl",kernelFunc:hg};function mg(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,filter:r}=e,{strides:a,dilations:c,pad:i,dimRoundingMode:l,inputShape:u}=s,d=$e(u,r.shape,a,c,i,l,!0),p=new pg(d);return t.runWebGLProgram(p,[o,r],"float32")}const xg={kernelName:kc,backendName:"webgl",kernelFunc:mg};class gg{constructor(e){this.variableNames=["X"],this.outputShape=[e,e],this.userCode=`
      void main() {
          ivec2 coords = getOutputCoords();
          float val = coords[0] == coords[1] ? getX(coords[0]) : 0.0;
          setOutput(val);
      }
    `}}function Cg(n){const{inputs:e,backend:t}=n,{x:s}=e,o=[...s.shape,...s.shape],r=E(s.shape),a=R({inputs:{x:s},backend:t,attrs:{shape:[r]}}),c=new gg(r),i=t.runWebGLProgram(c,[a],a.dtype),l=R({inputs:{x:i},backend:t,attrs:{shape:o}});return t.disposeIntermediateTensorInfo(a),t.disposeIntermediateTensorInfo(i),l}const $g={kernelName:Ac,backendName:"webgl",kernelFunc:Cg};class bg{constructor(e){this.variableNames=["x","W"],this.outputShape=e.outShape;const{inHeight:t,inWidth:s,padInfo:o,strideHeight:r,strideWidth:a,filterHeight:c,filterWidth:i,dilationHeight:l,dilationWidth:u}=e,{top:d,left:p}=o;this.userCode=`
      const ivec2 strides = ivec2(${r}, ${a});
      const ivec2 pads = ivec2(${d}, ${p});
      const float neg_infinity = -3.4e38;

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.w;
        ivec2 outTopLeftCorner =
            coords.yz * strides - pads;
        int hBeg = outTopLeftCorner.x;
        int wBeg = outTopLeftCorner.y;

        float curVal = neg_infinity;
        for (int h = 0; h < ${c}; h++) {
          int hIn = hBeg + h * ${l};

          if (hIn >= 0 && hIn < ${t}) {
            for (int w = 0; w < ${i}; w++) {
              int wIn = wBeg + w * ${u};

              if (wIn >= 0 && wIn < ${s}) {
                float xVal = getX(batch, hIn, wIn, d1);
                float wVal = getW(h, w, d1);

                float val = xVal + wVal;
                if (val > curVal) {
                  curVal = val;
                }
              }
            }
          }
        }

        float result = curVal;
        setOutput(result);
      }
    `}}function vg(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,filter:r}=e,{strides:a,pad:c,dilations:i}=s,l=Zs(o.shape,r.shape,a,c,"NHWC",i);let u;const d=new bg(l);u=t.runWebGLProgram(d,[o,r],"float32");const p=R({inputs:{x:u},backend:t,attrs:{shape:l.outShape}});return t.disposeIntermediateTensorInfo(u),p}const wg={kernelName:Oc,backendName:"webgl",kernelFunc:vg};function Ig(n){const{inputs:e,backend:t,attrs:s}=n,{equation:o}=s,r=e,{allDims:a,summedDims:c,idDims:i}=Vo(o,r.length);Mo(a.length,i,r);const{path:l,steps:u}=Wo(c,i),d=u.length;let p=null,h=a.length;const f=[];for(let g=0;g<d;++g){for(const x of u[g]){const{permutationIndices:m,expandDims:C}=Bo(h,i[x]);let $;Uo(m)?$=r[x]:($=H({inputs:{x:r[x]},backend:t,attrs:{perm:m}}),f.push($));const b=$.shape.slice();for(let v=0;v<C.length;++v)b.splice(C[v],0,1);q($.shape,b)||($=R({inputs:{x:$},backend:t,attrs:{shape:b}}),f.push($)),p===null?p=$:(p=fs({inputs:{a:$,b:p},backend:t}),f.push(p))}g<d-1&&(l[g]>=0&&(p=Zt({inputs:{x:p},backend:t,attrs:{axis:l[g]-(a.length-h),keepDims:!1}}),f.push(p)),h--)}for(const g of f)g!==p&&t.disposeIntermediateTensorInfo(g);return p}const Rg={kernelName:Dc,backendName:"webgl",kernelFunc:Ig};const yg="return (x >= 0.0) ? x : (exp(x) - 1.0);",Sg=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,Tg=D({opSnippet:yg,packedOpSnippet:Sg}),Eg={kernelName:Fc,backendName:"webgl",kernelFunc:Tg};const Ng="return (b >= 0.0) ? a : a * (b + 1.0);",kg=`
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`,Ag=n=>{const{inputs:e,backend:t}=n,{dy:s,y:o}=e,r=w().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new tt(kg,s.shape,o.shape):new Fe(Ng,s.shape,o.shape);return t.runWebGLProgram(r,[s,o],s.dtype)},Og={kernelName:Pc,backendName:"webgl",kernelFunc:Ag};const Dg=`
  return vec4(equal(a, b));
`,Fg="return float(a == b);",Pg=V({opSnippet:Fg,packedOpSnippet:Dg,dtype:"bool",cpuKernelImpl:uh}),_g={kernelName:In,backendName:"webgl",kernelFunc:Pg};const Lg=`
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  float p = ${Oo};
  float a1 = ${Do};
  float a2 = ${Fo};
  float a3 = ${Po};
  float a4 = ${_o};
  float a5 = ${Lo};

  float sign = sign(x);
  x = abs(x);
  float t = 1.0 / (1.0 + p * x);
  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));
`,Vg=D({opSnippet:Lg}),Bg={kernelName:_c,backendName:"webgl",kernelFunc:Vg};const Mg=nt+`
  return exp(x);
`,Wg=`
  vec4 result = exp(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,Qa=D({opSnippet:Mg,packedOpSnippet:Wg,cpuKernelImpl:dh,dtype:"float32"}),Ug={kernelName:Rn,backendName:"webgl",kernelFunc:Qa};function Cn(n){const{inputs:e,attrs:t,backend:s}=n,{dim:o}=t,{input:r}=e,a=r.shape.length,c=r.shape.slice();let i=o;return o<0&&(O(-(a+1)<=o,()=>`Axis must be in the interval [${-(a+1)}, ${a}]`),i=a+o+1),c.splice(i,0,1),R({inputs:{x:r},backend:s,attrs:{shape:c}})}const Gg={kernelName:Lc,backendName:"webgl",kernelFunc:Cn};const Hs="return exp(x) - 1.0;",zg=D({opSnippet:Hs,packedOpSnippet:Hs,cpuKernelImpl:ph}),Hg={kernelName:yn,backendName:"webgl",kernelFunc:zg};class Xs{constructor(e,t,s){this.variableNames=["real","imag"];const o=t[1];this.outputShape=t;const r=s?`2.0 * ${Math.PI}`:`-2.0 * ${Math.PI}`,a=s?`${o}.0`:"1.0";let c;if(e==="real")c="return real * expR - imag * expI;";else if(e==="imag")c="return real * expI + imag * expR;";else throw new Error(`FFT component must be either "real" or "imag", got ${e}.`);this.userCode=`
      const float exponentMultiplier = ${r};

      float unaryOpComplex(float real, float expR, float imag, float expI) {
        ${c}
      }

      float mulMatDFT(int batch, int index) {
        float indexRatio = float(index) / float(${o});
        float exponentMultiplierTimesIndexRatio =
            exponentMultiplier * indexRatio;

        float result = 0.0;

        for (int i = 0; i < ${o}; i++) {
          // x = (-2|2 * PI / N) * index * i;
          float x = exponentMultiplierTimesIndexRatio * float(i);
          float expR = cos(x);
          float expI = sin(x);
          float real = getReal(batch, i);
          float imag = getImag(batch, i);

          result +=
              unaryOpComplex(real, expR, imag, expI) / ${a};
        }

        return result;
      }

      void main() {
        ivec2 coords = getOutputCoords();
        setOutput(mulMatDFT(coords[0], coords[1]));
      }
    `}}function Za(n,e,t){const s=t.texData.get(n.dataId),o=E(n.shape),r=n.shape[n.shape.length-1],a=o/r,c=R({inputs:{x:n},backend:t,attrs:{shape:[a,r]}}),i=c.shape,l=new Xs("real",i,e),u=new Xs("imag",i,e),d=[{dataId:s.complexTensorInfos.real.dataId,dtype:s.complexTensorInfos.real.dtype,shape:i},{dataId:s.complexTensorInfos.imag.dataId,dtype:s.complexTensorInfos.imag.dtype,shape:i}],p=t.runWebGLProgram(l,d,"float32"),h=t.runWebGLProgram(u,d,"float32"),f=Te({inputs:{real:p,imag:h},backend:t});t.disposeIntermediateTensorInfo(p),t.disposeIntermediateTensorInfo(h);const g=R({inputs:{x:f},backend:t,attrs:{shape:n.shape}});return t.disposeIntermediateTensorInfo(c),t.disposeIntermediateTensorInfo(f),g}function Xg(n){const{inputs:e,backend:t}=n,{input:s}=e;return Za(s,!1,t)}const Kg={kernelName:Vc,backendName:"webgl",kernelFunc:Xg};class jg{constructor(e,t){this.outputShape=[],this.customUniforms=[{name:"value",type:"float"}],this.variableNames=["x"],this.outputShape=e,this.userCode=`
      void main() {
        // Input can be obtained from uniform value.
        setOutput(value);
      }
    `}}function St(n){const{backend:e,attrs:t}=n,{shape:s,value:o}=t;let{dtype:r}=t;if(r=r||Mc(o),r==="string"){const a=B(r,E(s));return a.fill(o),e.makeTensorInfo(s,r,a)}else{const a=new jg(s,o),c=[[o]];return e.runWebGLProgram(a,[],r,c)}}const qg={kernelName:Bc,backendName:"webgl",kernelFunc:St};class Yg{constructor(e){this.variableNames=["Image"],this.outputShape=[];const t=e[2];this.outputShape=e,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];

          int coordX = ${t} - x - 1;
          float outputValue;
          if(coordX >= 0 && coordX < ${t}) {
            outputValue = getImage(coords[0], coords[1], coordX, coords[3]);
          } else {
            outputValue = getImage(coords[0], coords[1], coords[2], coords[3]);
          }
          setOutput(outputValue);
        }
    `}}const Qg={kernelName:Wc,backendName:"webgl",kernelFunc:({inputs:n,backend:e})=>{const{image:t}=n,s=e,o=new Yg(t.shape);return s.runWebGLProgram(o,[t],t.dtype)}};const Ks="return floor(x);",Zg=D({opSnippet:Ks,packedOpSnippet:Ks,cpuKernelImpl:hh}),Jg={kernelName:Sn,backendName:"webgl",kernelFunc:Zg};const eC=`
  float s = sign(a) * sign(b);
  int ia = round(a);
  int ib = round(b);
  if (ib != 0) {
    // Windows (D3D) wants guaranteed non-zero int division at compile-time.
    return float(idiv(ia, ib, s));
  } else {
    return NAN;
  }
`,tC=`
  ivec4 ia = round(a);
  ivec4 ib = round(b);
  bvec4 cond = notEqual(ib, ivec4(0));
  ivec4 result = ivec4(0);
  vec4 s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    result[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    result[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    result[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    result[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4(result);
`,nC=V({opSnippet:eC,packedOpSnippet:tC,dtype:"int32"}),sC={kernelName:Tn,backendName:"webgl",kernelFunc:nC};class oC{constructor(e){this.variableNames=["A"];const t=K(),[s,o]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${o}.0, ${s}.0);

        vec4 values = ${t.texture2D}(A, uv);
        float value;
        if (depth == 0) {
          value = values.r;
        } else if (depth == 1) {
          value = values.g;
        } else if (depth == 2) {
          value = values.b;
        } else if (depth == 3) {
          value = values.a;
        }

        setOutput(floor(value * 255.0 + 0.5));
      }
    `}}class rC{constructor(e){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0;const t=K(),[s,o]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];

        vec4 result = vec4(0.);

        for(int row=0; row<=1; row++) {
          for(int col=0; col<=1; col++) {
            texC = coords[1] + row;
            depth = coords[2] + col;

            vec2 uv = (vec2(texC, texR) + halfCR) /
                       vec2(${o}.0, ${s}.0);
            vec4 values = ${t.texture2D}(A, uv);
            float value;
            if (depth == 0) {
              value = values.r;
            } else if (depth == 1) {
              value = values.g;
            } else if (depth == 2) {
              value = values.b;
            } else if (depth == 3) {
              value = values.a;
            }

            result[row * 2 + col] = floor(value * 255.0 + 0.5);
          }
        }

        ${t.output} = result;
      }
    `}}const aC={kernelName:Uc,backendName:"webgl",kernelFunc:iC};let Ge,sn=w().getBool("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU");function iC(n){const{inputs:e,backend:t,attrs:s}=n;let{pixels:o}=e;const{numChannels:r}=s,a=typeof HTMLVideoElement<"u"&&o instanceof HTMLVideoElement,c=typeof HTMLImageElement<"u"&&o instanceof HTMLImageElement,[i,l]=a?[o.videoWidth,o.videoHeight]:[o.width,o.height],u=[l,i],d=[l,i,r];if(c||a){const g=w().getBool("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU");(Ge==null||g!==sn)&&(sn=g,Ge=document.createElement("canvas").getContext("2d",{willReadFrequently:sn})),Ge.canvas.width=i,Ge.canvas.height=l,Ge.drawImage(o,0,0,i,l),o=Ge.canvas}const p=t.makeTensorInfo(u,"int32");t.texData.get(p.dataId).usage=te.PIXELS,t.gpgpu.uploadPixelDataToTexture(t.getTexture(p.dataId),o);const h=w().getBool("WEBGL_PACK")?new rC(d):new oC(d),f=t.runWebGLProgram(h,[p],"int32");return t.disposeData(p.dataId),f}function cC(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,filter:r,bias:a,preluActivationWeights:c}=e,{strides:i,pad:l,dataFormat:u,dilations:d,dimRoundingMode:p,activation:h,leakyreluAlpha:f}=s,g=bt(u),x=$e(o.shape,r.shape,i,d,l,p,!1,g);let m;const C=[],$=a!=null,b=c!=null,v=h==="leakyrelu",T=()=>{const I=[o,r],A=(k,F)=>{if(F==="NCHW"&&k.shape.length===1&&k.shape[0]!==1){const P=R({inputs:{x:k},backend:t,attrs:{shape:[k.shape[0],1,1]}});return C.push(P),P}return k};if($&&I.push(A(a,u)),b&&I.push(A(c,u)),v){const k=t.makeTensorInfo([],"float32",je(f,"float32"));I.push(k),C.push(k)}return I};if(x.filterHeight===1&&x.filterWidth===1&&x.dilationHeight===1&&x.dilationWidth===1&&x.strideHeight===1&&x.strideWidth===1&&(x.padInfo.type==="SAME"||x.padInfo.type==="VALID"))m=Xa({x:o,filter:r,convInfo:x,backend:t,bias:a,activation:h,preluActivationWeights:c,leakyreluAlpha:f});else if(x.strideWidth<=2&&g==="channelsLast"&&w().getBool("WEBGL_EXP_CONV")){const I=h?xt(h,!0):null,A=new Ha(x,$,I,b,v),k=[[x.padInfo.top,x.padInfo.left],[x.strideHeight,x.strideWidth],[x.dilationHeight,x.dilationWidth],[x.inHeight,x.inWidth]],F=T();m=t.runWebGLProgram(A,F,"float32",k)}else if(w().getBool("WEBGL_CONV_IM2COL"))m=Ka({x:o,filter:r,convInfo:x,backend:t,bias:a,activation:h,preluActivationWeights:c,leakyreluAlpha:f});else{const I=h?xt(h,!1):null,A=new za(x,$,I,b,v),k=T();m=t.runWebGLProgram(A,k,"float32")}const S=R({inputs:{x:m},backend:t,attrs:{shape:x.outShape}});return C.push(m),C.forEach(I=>t.disposeIntermediateTensorInfo(I)),S}const lC={kernelName:Gc,backendName:"webgl",kernelFunc:cC};function uC(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,filter:r,bias:a,preluActivationWeights:c}=e,{strides:i,pad:l,dilations:u,dimRoundingMode:d,activation:p,leakyreluAlpha:h}=s,f=[];let g=u;g==null&&(g=[1,1]),O(Ke(i,g),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${i} and dilations '${g}'`);const x=$e(o.shape,r.shape,i,g,l,d,!0),m=w().getBool("WEBGL_PACK_DEPTHWISECONV")&&x.strideWidth<=2&&x.outChannels/x.inChannels===1,C=p?xt(p,m):null,$=[o,r],b=a!=null,v=c!=null,T=p==="leakyrelu";if(b&&$.push(a),v&&$.push(c),T){const k=t.makeTensorInfo([],"float32",je(h,"float32"));$.push(k),f.push(k)}let S;m?S=new Ya(x,b,C,v,T):S=new qa(x,b,C,v,T);const I=[[x.padInfo.top,x.padInfo.left],[x.strideHeight,x.strideWidth],[x.dilationHeight,x.dilationWidth],[x.inHeight,x.inWidth]],A=t.runWebGLProgram(S,$,"float32",I);return f.forEach(k=>t.disposeIntermediateTensorInfo(k)),A}const dC={kernelName:zc,backendName:"webgl",kernelFunc:uC};class pC{constructor(e,t,s,o){this.sliceDim=e,this.strides=t,this.paramsShape=o,this.variableNames=["x","indices"],this.outputShape=s;const r=_(s.length);let a=`
    int index;`;for(let c=0;c<this.sliceDim;c++)a+=`
          index = round(getIndices(coords[0], ${c}));
          out_of_bounds = out_of_bounds || index < 0;
          out_of_bounds = out_of_bounds || index >= ${this.paramsShape[c]};
          flattenIndex += index * ${this.strides[c]};`;this.userCode=`
         void main() {
          ${r} coords = getOutputCoords();
          int flattenIndex = 0;
          bool out_of_bounds = false;

          ${a}

          setOutput(out_of_bounds ? 0.0 : getX(flattenIndex, coords[1]));
        }
      `}}function hC(n){const{inputs:e,backend:t}=n,{params:s,indices:o}=e,r=o.shape,a=r[r.length-1],c=E(s.shape),[i,l,u,d]=Hn(s,o),p=R({inputs:{x:o},backend:t,attrs:{shape:[l,a]}}),h=R({inputs:{x:s},backend:t,attrs:{shape:[E(s.shape)/u,u]}});if(t.shouldExecuteOnCPU([s,o])||s.dtype==="string"){const m=t.readSync(o.dataId),C=t.bufferSync(s),$=fh(m,C,s.dtype,l,a,u,d,s.shape,c);return t.makeTensorInfo(i,s.dtype,$.values)}const f=new pC(a,d,[l,u],s.shape),g=t.runWebGLProgram(f,[h,p],h.dtype),x=R({inputs:{x:g},backend:t,attrs:{shape:i}});return t.disposeIntermediateTensorInfo(p),t.disposeIntermediateTensorInfo(h),t.disposeIntermediateTensorInfo(g),x}const fC={kernelName:Hc,backendName:"webgl",kernelFunc:hC};class mC{constructor(e,t){this.variableNames=["A","indices"],this.outputShape=t,this.rank=t.length;const s=_(this.rank),o=xC(e);this.userCode=`
      void main() {
        ${s} resRC = getOutputCoords();
        int index = int(getIndices(resRC.x, resRC.z));
        float inBounds = (index >= 0) && (index < ${e[2]}) ? 1.0 : 0.0;
        setOutput(inBounds * getA(${o}));
      }
    `}}function xC(n,e){const t=["resRC.x","resRC.y","resRC.z","resRC.w"],s=[];for(let o=0;o<n.length;o++)o===2?s.push("index"):s.push(`${t[o]}`);return s.join()}function Ja(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,indices:r}=e,{axis:a,batchDims:c}=s,i=X(a,o.shape)[0];if(w().get("DEBUG")){const C=t.readSync(r.dataId),$=o.shape[i];for(let b=0;b<C.length;++b){const v=C[b];O(v<=$-1&&v>=0,()=>`GatherV2: the index value ${v} is not in [0, ${$-1}]`)}}const l=sr(o,r,i,c),u=E(r.shape),d=[],p=R({inputs:{x:o},backend:t,attrs:{shape:[l.batchSize,l.outerSize,l.dimSize,l.sliceSize]}}),h=R({inputs:{x:r},backend:t,attrs:{shape:[l.batchSize,u/l.batchSize]}});d.push(p),d.push(h);const f=[l.batchSize,l.outerSize,u/l.batchSize,l.sliceSize];if(t.shouldExecuteOnCPU([o,r])||o.dtype==="string"){const C=t.bufferSync(h),$=t.bufferSync(p),b=mh($,C,f);return d.forEach(v=>t.disposeIntermediateTensorInfo(v)),t.makeTensorInfo(l.outputShape,b.dtype,b.values)}const g=new mC(p.shape,f),x=t.runWebGLProgram(g,[p,h],p.dtype);d.push(x);const m=R({inputs:{x},backend:t,attrs:{shape:l.outputShape}});return d.forEach(C=>t.disposeIntermediateTensorInfo(C)),m}const gC={kernelName:Xc,backendName:"webgl",kernelFunc:Ja};const CC="return float(a > b);",$C=`
  return vec4(greaterThan(a, b));
`,bC=V({opSnippet:CC,packedOpSnippet:$C,cpuKernelImpl:xh,dtype:"bool"}),vC={kernelName:En,backendName:"webgl",kernelFunc:bC};const wC="return float(a >= b);",IC=`
  return vec4(greaterThanEqual(a, b));
`,RC=V({opSnippet:wC,packedOpSnippet:IC,dtype:"bool",cpuKernelImpl:gh}),yC={kernelName:Nn,backendName:"webgl",kernelFunc:RC};function SC(n){const{inputs:e,backend:t}=n,{input:s}=e;return Za(s,!0,t)}const TC={kernelName:Kc,backendName:"webgl",kernelFunc:SC};const EC="return float(!isnan(x) && !isinf(x));",NC=D({opSnippet:EC,dtype:"bool"}),kC={kernelName:jc,backendName:"webgl",kernelFunc:NC};const AC="return float(isinf(x));",OC=D({opSnippet:AC,dtype:"bool"}),DC={kernelName:qc,backendName:"webgl",kernelFunc:OC};const FC="return float(isnan(x));",PC=D({opSnippet:FC,dtype:"bool"}),_C={kernelName:Yc,backendName:"webgl",kernelFunc:PC};const LC="return float(a < b);",VC=`
  return vec4(lessThan(a, b));
`,BC=V({opSnippet:LC,packedOpSnippet:VC,cpuKernelImpl:Ch,dtype:"bool"}),MC={kernelName:kn,backendName:"webgl",kernelFunc:BC};const WC="return float(a <= b);",UC=`
  return vec4(lessThanEqual(a, b));
`,GC=V({opSnippet:WC,packedOpSnippet:UC,cpuKernelImpl:$h,dtype:"bool"}),zC={kernelName:An,backendName:"webgl",kernelFunc:GC};function HC(n){const{backend:e,attrs:t}=n,{start:s,stop:o,num:r}=t,a=bh(s,o,r);return e.makeTensorInfo([a.length],"float32",a)}const XC={kernelName:Qc,backendName:"webgl",kernelFunc:HC};const KC=nt+`
  return x < 0.0 ? 0./0. : log(x);
`,jC=`
  vec4 result = log(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : (x.r < 0.0 ? 0./0. : result.r);
  result.g = isNaN.g ? x.g : (x.g < 0.0 ? 0./0. : result.g);
  result.b = isNaN.b ? x.b : (x.b < 0.0 ? 0./0. : result.b);
  result.a = isNaN.a ? x.a : (x.a < 0.0 ? 0./0. : result.a);
  return result;
`,qC=D({opSnippet:KC,packedOpSnippet:jC,cpuKernelImpl:vh}),YC={kernelName:On,backendName:"webgl",kernelFunc:qC};const QC=nt+`
  return log(1.0 + x);
`,ZC=D({opSnippet:QC}),JC={kernelName:Zc,backendName:"webgl",kernelFunc:ZC};const e0="return float(a >= 1.0 && b >= 1.0);",t0=`
  return vec4(
    vec4(greaterThanEqual(a, vec4(1.0))) *
    vec4(greaterThanEqual(b, vec4(1.0))));
`,n0=V({opSnippet:e0,packedOpSnippet:t0,dtype:"bool"}),s0={kernelName:Jc,backendName:"webgl",kernelFunc:n0};const o0="return float(!(x >= 1.0));",r0=D({opSnippet:o0}),a0={kernelName:el,backendName:"webgl",kernelFunc:r0};const i0="return float(a >= 1.0 || b >= 1.0);",c0=`
  return min(
    vec4(greaterThanEqual(a, vec4(1.0))) +
    vec4(greaterThanEqual(b, vec4(1.0))),
    vec4(1.0));
`,l0=V({opSnippet:i0,packedOpSnippet:c0,dtype:"bool"}),u0={kernelName:tl,backendName:"webgl",kernelFunc:l0};class d0{constructor(e,t,s,o,r){this.variableNames=["x"],this.outputShape=[];const a=t,c=e[3]-1;this.outputShape=e;let i;const l=`float(${s}) + float(${o}) * sum`;r===.5?i=`inversesqrt(${l})`:r===1?i=`1.0/(${l})`:i=`exp(log(${l}) * float(-${r}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];
        int d = coords[3];
        float x = getX(b, r, c, d);
        float sum = 0.0;
        for (int j = -${a}; j <= ${a}; j++) {
          int idx = d + j;
          if (idx >= 0 && idx <=  ${c}) {
            float z = getX(b, r, c, idx);
            sum += z * z;
          }
        }
        float val = x * ${i};
        setOutput(val);
      }
    `}}class p0{constructor(e,t,s,o,r){this.variableNames=["x"],this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0;const a=t,c=e[3]-1;this.outputShape=e;let i;const l=`float(${s}) + float(${o}) * sum`;r===.5?i=`inversesqrt(${l})`:r===1?i=`1.0/(${l})`:i=`exp(log(${l}) * float(-${r}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords.x;
        int r = coords.y;
        int c = coords.z;
        int d = coords.w;

        bool hasNextCol = d < ${this.outputShape[3]};
        bool hasNextRow = c < ${this.outputShape[2]};

        vec4 sum = vec4(0.);
        vec4 xFragAtOutputCoords = getX(b, r, c, d);

        vec4 xAtOutputCoords = vec4(
          getChannel(xFragAtOutputCoords, vec2(c, d)),
          hasNextCol ?
            getChannel(xFragAtOutputCoords, vec2(c, d + 1)) : 0.0,
          hasNextRow ?
            getChannel(xFragAtOutputCoords , vec2(c + 1, d)) : 0.0,
          (hasNextRow && hasNextCol) ?
            getChannel(xFragAtOutputCoords, vec2(c + 1, d + 1)) : 0.0
        );

        int firstChannel = d - ${a};
        vec2 cache = vec2(0.);
        if(firstChannel >= 0){
          vec4 firstChannelFrag = getX(b, r, c, firstChannel);
          cache.x = getChannel(firstChannelFrag, vec2(c, firstChannel));
            if(hasNextRow){
              cache.y = getChannel(firstChannelFrag, vec2(c + 1, firstChannel));
            }
        }

        ivec2 depth = ivec2(d, d + 1);
        for (int j = - ${a}; j <= ${a}; j++) {
          ivec2 idx = depth + j;
          bvec2 aboveLowerBound = greaterThanEqual(idx, ivec2(0));
          bvec2 belowUpperBound = lessThanEqual(idx, ivec2(${c}));

          bool depthInRange = aboveLowerBound.x && belowUpperBound.x;
          bool depthPlusOneInRange = aboveLowerBound.y && belowUpperBound.y;

          if(depthInRange || depthPlusOneInRange){
            vec4 z = vec4(0.);
            vec4 xFragAtCurrentDepth;
            z.xz = cache.xy;
            if(depthPlusOneInRange && hasNextCol){
              xFragAtCurrentDepth = idx.y != d ?
                getX(b, r, c, idx.y) : xFragAtOutputCoords;
              z.y = getChannel(xFragAtCurrentDepth, vec2(c, idx.y));
              if(hasNextRow){
                z.w = getChannel(xFragAtCurrentDepth, vec2(c + 1, idx.y));
              }
            }
            cache.xy = z.yw;
            sum += z * z;
          }
        }
        vec4 result = xAtOutputCoords * ${i};
        setOutput(result);
      }
    `}}const h0=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{depthRadius:r,bias:a,alpha:c,beta:i}=s,l=w().getBool("WEBGL_PACK_NORMALIZATION")?new p0(o.shape,r,a,c,i):new d0(o.shape,r,a,c,i);return t.runWebGLProgram(l,[o],o.dtype)},f0={kernelName:nl,backendName:"webgl",kernelFunc:h0};class m0{constructor(e,t,s,o,r){this.variableNames=["inputImage","outputImage","dy"],this.outputShape=[],this.outputShape=e,this.depth=e[3],this.depthRadius=t,this.bias=s,this.alpha=o,this.beta=r,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];

        float result = 0.0;
        for (int d = 0; d < ${this.depth}; ++d) {
          int depthBegin = int(max(0.0, float(d - ${t})));
          int depthEnd = int(min(float(${this.depth}),
              float(d + ${t} + 1)));

          const int MIN_DEPTH_BEGIN = 0;
          const int MAX_DEPTH_END = ${this.depth};

          float norm = 0.0;
          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            }
            else {
              break;
            }
          }

          norm = float(${o}) * norm + float(${s});

          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd){
              float dyi = -2.0 * float(${o})
                * float(${r})
                * getInputImage(b, r, c, k) * getOutputImage(b, r, c, d)
                / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * ${r});
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            }
            else {
              break;
            }
          }
      }
      setOutput(result);
      }
    `}}const x0=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:o,y:r,dy:a}=e,{depthRadius:c,bias:i,alpha:l,beta:u}=s,d=new m0(o.shape,c,i,l,u);return t.runWebGLProgram(d,[o,r,a],o.dtype)},g0={kernelName:sl,backendName:"webgl",kernelFunc:x0};function C0(n,e,t,s){const o=E(e),a=E(n.shape)/o,c=R({inputs:{x:n},attrs:{shape:[a,o]},backend:s}),i=Me(c,n.dtype,"max",s),l=R({inputs:{x:i},attrs:{shape:t},backend:s});return s.disposeIntermediateTensorInfo(c),s.disposeIntermediateTensorInfo(i),l}function ei(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{reductionIndices:r,keepDims:a}=s,c=o.shape.length,i=X(r,o.shape);let l=i;const u=se(l,c),d=u!=null,p=t.shouldExecuteOnCPU([o]);let h=o;if(d){if(p){const $=t.texData.get(h.dataId).values,b=new Array(c);for(let S=0;S<b.length;S++)b[S]=o.shape[u[S]];const v=ps($,o.shape,o.dtype,u,b);h=t.makeTensorInfo(b,o.dtype);const T=t.texData.get(h.dataId);T.values=v}else h=Qt(o,u,t);l=oe(l.length,c)}de("max",l,c);const[f,g]=fe(h.shape,l);let x=f;a&&(x=me(f,i));let m;if(p){const $=t.texData.get(h.dataId).values,b=wh($,E(g),x,o.dtype);m=t.makeTensorInfo(x,o.dtype);const v=t.texData.get(m.dataId);v.values=b}else m=C0(h,g,x,t);return d&&t.disposeIntermediateTensorInfo(h),m}const $0={kernelName:ol,backendName:"webgl",kernelFunc:ei};const b0=hs+`
  return max(a, b);
`,v0=`
  vec4 result = vec4(max(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Be+`
  return result;
`,w0=V({opSnippet:b0,packedOpSnippet:v0,cpuKernelImpl:Ih}),I0={kernelName:Dn,backendName:"webgl",kernelFunc:w0};function R0(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e;Ye(o,"maxPool");const{filterSize:r,strides:a,pad:c,dimRoundingMode:i}=s,l=1;O(Ke(a,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${a} and dilations '${l}'`);const u=Xe(o.shape,r,a,l,c,i);if(u.filterWidth===1&&u.filterHeight===1&&q(u.inShape,u.outShape))return Z({inputs:{x:o},backend:t});const d=new gt(u,"max",!1);return t.runWebGLProgram(d,[o],o.dtype)}const y0={kernelName:rl,backendName:"webgl",kernelFunc:R0};function S0(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{filterSize:r,strides:a,pad:c,dataFormat:i,dimRoundingMode:l}=s,u=[1,1,1],d=$t(o.shape,r,a,u,c,l,i),p=new ms(d,"max",!1);return t.runWebGLProgram(p,[o],o.dtype)}const T0={kernelName:al,backendName:"webgl",kernelFunc:S0};class E0{constructor(e){this.variableNames=["dy","maxPos"],this.outputShape=e.inShape;const t=e.strideHeight,s=e.strideWidth,o=e.dilationHeight,r=e.effectiveFilterHeight,a=e.effectiveFilterWidth,c=r-1-e.padInfo.top,i=a-1-e.padInfo.left,l=r*a-1;this.userCode=`
      const ivec2 pads = ivec2(${c}, ${i});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${r};
          wR += ${o}) {
          float dyR = float(dyRCorner + wR) / ${t}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${a}; wC++) {
            float dyC = float(dyCCorner + wC) / ${s}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);
            int maxPosValue = ${l} - int(getMaxPos(b, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            int curPosValue = wR * ${a} + wC;
            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

            dotProd += dyValue * mask;
          }
        }
        setOutput(dotProd);
      }
    `}}class N0{constructor(e){this.variableNames=["dy","maxPos"],this.outputShape=e.inShape;const t=e.strideDepth,s=e.strideHeight,o=e.strideWidth,r=e.dilationDepth,a=e.dilationHeight,c=e.dilationWidth,i=e.effectiveFilterDepth,l=e.effectiveFilterHeight,u=e.effectiveFilterWidth,d=i-1-e.padInfo.front,p=l-1-e.padInfo.top,h=u-1-e.padInfo.left,f=i*l*u-1;this.userCode=`
      const ivec3 pads = ivec3(${d}, ${p}, ${h});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${i};
           wD += ${r}) {
          float dyD = float(dyDCorner + wD) / ${t}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${l};
              wR += ${a}) {
            float dyR = float(dyRCorner + wR) / ${s}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${u};
                wC += ${c}) {
              float dyC = float(dyCCorner + wC) / ${o}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);
              int maxPosValue = ${f} -
                  int(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              int curPosValue =
                  wD * ${l} * ${u} +
                  wR * ${u} + wC;
              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

              dotProd += dyValue * mask;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}function k0(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,input:r}=e,a=r,{filterSize:c,strides:i,pad:l,dimRoundingMode:u}=s,d=[1,1,1],p=$t(a.shape,c,i,d,l,u),h=new ms(p,"max",!0),f=t.runWebGLProgram(h,[a],a.dtype),g=new N0(p),x=t.runWebGLProgram(g,[o,f],a.dtype);return t.disposeIntermediateTensorInfo(f),x}const A0={kernelName:il,backendName:"webgl",kernelFunc:k0};function O0(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,input:r,output:a}=e,c=r;Ye([r,a],"maxPoolGrad");const{filterSize:i,strides:l,pad:u,dimRoundingMode:d}=s,p=Xe(c.shape,i,l,1,u,d),h=!0,f=new gt(p,"max",h),g=t.runWebGLProgram(f,[c],c.dtype),x=new E0(p),m=t.runWebGLProgram(x,[o,g],c.dtype);return t.disposeIntermediateTensorInfo(g),m}const D0={kernelName:cl,backendName:"webgl",kernelFunc:O0};function F0(n,e,t,s){let o=new gt(t,"max",!1);const r=s.runWebGLProgram(o,[n],"float32");o=new gt(t,"max",!0,!0,e);const a=s.runWebGLProgram(o,[n],"float32");return[r,a]}const P0={kernelName:ll,backendName:"webgl",kernelFunc:({inputs:n,attrs:e,backend:t})=>{const{x:s}=n,{filterSize:o,strides:r,pad:a,includeBatchInIndex:c}=e,i=t;O(s.shape.length===4,()=>`Error in maxPool: input must be rank 4 but got rank ${s.shape.length}.`);const l=[1,1];O(Ke(r,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${r} and dilations '${l}'`);const u=Xe(s.shape,o,r,l,a),[d,p]=F0(s,c,u,i);return[d,p]}};function _0(n,e,t,s){const o=E(e),a=E(n.shape)/o,c=R({inputs:{x:n},attrs:{shape:[a,o]},backend:s}),i=Me(c,"float32","mean",s),l=R({inputs:{x:i},attrs:{shape:t},backend:s});return s.disposeIntermediateTensorInfo(c),s.disposeIntermediateTensorInfo(i),l}const L0={kernelName:ul,backendName:"webgl",kernelFunc:({inputs:n,attrs:e,backend:t})=>{const{x:s}=n,{keepDims:o,axis:r}=e,a=t,c=s.shape.length,i=X(r,s.shape);let l=i;const u=se(l,c),d=u!=null,p=a.shouldExecuteOnCPU([s]),h=[];let f=s;if(d){if(p){const b=a.texData.get(f.dataId).values,v=new Array(c);for(let I=0;I<v.length;I++)v[I]=s.shape[u[I]];const T=ps(b,s.shape,s.dtype,u,v);f=a.makeTensorInfo(v,s.dtype);const S=a.texData.get(f.dataId);S.values=T}else f=Qt(s,u,a);h.push(f),l=oe(l.length,c)}de("sum",l,c);const[g,x]=fe(f.shape,l);let m=g;o&&(m=me(g,i));const C=_0(f,x,m,a);for(const $ of h)a.disposeIntermediateTensorInfo($);return C}};function V0(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,keepDims:a}=s,c=o.shape.length,i=X(r,o.shape);let l=i;const u=se(l,c);let d=o;u!=null&&(d=H({inputs:{x:o},backend:t,attrs:{perm:u}}),l=oe(l.length,o.shape.length)),de("min",l,c);const[p,h]=fe(d.shape,l),f=E(h),g=R({inputs:{x:d},backend:t,attrs:{shape:[-1,f]}}),x=Me(g,g.dtype,"min",t);let m;if(a){const C=me(p,i);m=R({inputs:{x},backend:t,attrs:{shape:C}})}else m=R({inputs:{x},backend:t,attrs:{shape:p}});return t.disposeIntermediateTensorInfo(g),t.disposeIntermediateTensorInfo(x),u!=null&&t.disposeIntermediateTensorInfo(d),m}const B0={kernelName:dl,backendName:"webgl",kernelFunc:V0};const M0=hs+`
  return min(a, b);
`,W0=`
  vec4 result = vec4(min(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Be+`
  return result;
`,U0=V({opSnippet:M0,packedOpSnippet:W0,cpuKernelImpl:Rh}),G0={kernelName:Fn,backendName:"webgl",kernelFunc:U0};class z0{constructor(e,t,s){this.variableNames=["x"],this.outputShape=t.map((u,d)=>u[0]+e[d]+u[1]);const o=e.length,r=_(o),a=t.map(u=>u[0]).join(","),c=t.map((u,d)=>u[0]+e[d]).join(","),i=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,o),l=s==="reflect"?0:1;if(o===1){this.userCode=`
        int start = ${a};
        int end = ${c};

        void main() {
          int outC = getOutputCoords();
          if (outC < start) {
            outC = start * 2 - outC - ${l};
          } else if(outC >= end) {
            outC = (end - 1) * 2 - outC + ${l};
          }
          setOutput(getX(outC - start));
        }
      `;return}this.userCode=`
      ${r} start = ${r}(${a});
      ${r} end = ${r}(${c});

      void main() {
        ${r} outC = getOutputCoords();
        for (int i = 0; i < ${o}; i++) {
          if (outC[i] < start[i]) {
            outC[i] = start[i] * 2 - outC[i] - ${l};
          } else if(outC[i] >= end[i]) {
            outC[i] = (end[i] - 1) * 2 - outC[i] + ${l};
          }
        }
        ${r} coords = outC - start;
        setOutput(getX(${i}));
      }
    `}}class H0{constructor(e,t,s){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t.map((f,g)=>f[0]+e[g]+f[1]);const o=e.length,r=_(o),a=t.map(f=>f[0]).join(","),c=t.map((f,g)=>f[0]+e[g]).join(","),i=G("rc",o),l=G("source",o),u=`${i[o-1]} < ${this.outputShape[o-1]}`,d=o===1?"source":`vec2(${l.slice(-2).join()})`,p=s==="reflect"?0:1;let h="";if(o===1){const f=`
        ${r} source = rc;
        if (source < start) {
          source = start * 2 - source - ${p};
        } else if (source >= end) {
          source = (end - 1) * 2 - source + ${p};
        }
        source -= start;
      `;h=`
        ${r} rc = outputLoc;
        ${f}
        result[0] = getChannel(getX(${l.join()}), ${d});
        ${i[o-1]} += 1;
        if(${u}) {
          ${f}
          result[1] = getChannel(getX(${l.join()}), ${d});
        }
      `}else{const f=`
        ${r} source = rc;
        ${r} lt = ${r}(lessThan(source, start));
        ${r} gte = ${r}(greaterThanEqual(source, end));
        ${r} orig = 1 - (lt + gte);
        source = orig * source +
                lt * (start * 2 - source - ${p}) +
                gte * ((end - 1) * 2 - source + ${p});
        source -= start;
      `;h=`
        ${r} rc = outputLoc;
        ${f}
        result[0] = getChannel(getX(${l.join()}), ${d});
        ${i[o-1]} += 1;
        if(${u}) {
          ${f}
          result[1] = getChannel(getX(${l.join()}), ${d});
        }
        rc = outputLoc;
        ${i[o-2]} += 1;
        if(${i[o-2]} < ${this.outputShape[o-2]}) {
          ${f}
          result[2] = getChannel(getX(${l.join()}), ${d});
          ${i[o-1]} += 1;
          if(${u}) {
            ${f}
            result[3] = getChannel(getX(${l.join()}), ${d});
          }
        }
      `}this.userCode=`
      const ${r} start = ${r}(${a});
      const ${r} end = ${r}(${c});

      void main() {
        ${r} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${h}
        setOutput(result);
      }
    `}}const X0=({inputs:n,backend:e,attrs:t})=>{const{x:s}=n,{paddings:o,mode:r}=t,a=w().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new H0(s.shape,o,r):new z0(s.shape,o,r);return e.runWebGLProgram(a,[s],s.dtype)},K0={kernelName:pl,backendName:"webgl",kernelFunc:X0};const j0=`if (b == 0.0) return NAN;
  return mod(a, b);`,q0=`
  vec4 result = mod(a, b);
  bvec4 isNaN = equal(b, vec4(0.0));
  `+Be+`
  return result;
`,Y0=V({opSnippet:j0,packedOpSnippet:q0}),Q0={kernelName:hl,backendName:"webgl",kernelFunc:Y0};class Z0{constructor(e,t,s){this.variableNames=["probs"],this.customUniforms=[{name:"seed",type:"float"}],this.outputShape=[e,s],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];

        float r = random(seed);
        float cdf = 0.0;

        for (int i = 0; i < ${t-1}; i++) {
          cdf += getProbs(batch, i);

          if (r < cdf) {
            setOutput(float(i));
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutput(float(${t-1}));
      }
    `}}const J0=`
if (a == b) {
  return 1.0;
};
return a / b;`,e$=`
  // vec4 one = vec4(equal(a, b));
  // return one + (vec4(1.0) - one) * a / b;
  vec4 result = a / b;
  if(a.x == b.x) {
    result.x = 1.;
  }
  if(a.y == b.y) {
    result.y = 1.;
  }
  if(a.z == b.z) {
    result.z = 1.;
  }
  if(a.w == b.w) {
    result.w = 1.;
  }

  return result;
`,ti=V({opSnippet:J0,packedOpSnippet:e$,checkOutOfBounds:!0}),t$={kernelName:fl,backendName:"webgl",kernelFunc:ti};const js="return a - b;",ni=V({opSnippet:js,packedOpSnippet:js,supportsComplex:!0,cpuKernelImpl:Hh}),n$={kernelName:Un,backendName:"webgl",kernelFunc:ni};function si(n){const{inputs:e,backend:t,attrs:s}=n,{logits:o}=e,{dim:r}=s,a=X([r],o.shape),c=ei({inputs:{x:o},backend:t,attrs:{reductionIndices:a,keepDims:!1}}),i=me(c.shape,a),l=R({inputs:{x:c},backend:t,attrs:{shape:i}}),u=ni({inputs:{a:o,b:l},backend:t}),d=Qa({inputs:{x:u},backend:t}),p=Zt({inputs:{x:d},backend:t,attrs:{axis:a,keepDims:!1}}),h=R({inputs:{x:p},backend:t,attrs:{shape:i}}),f=ti({inputs:{a:d,b:h},backend:t});return t.disposeIntermediateTensorInfo(c),t.disposeIntermediateTensorInfo(l),t.disposeIntermediateTensorInfo(u),t.disposeIntermediateTensorInfo(d),t.disposeIntermediateTensorInfo(p),t.disposeIntermediateTensorInfo(h),f}const s$={kernelName:ml,backendName:"webgl",kernelFunc:si};function o$(n){const{inputs:e,backend:t,attrs:s}=n,{logits:o}=e,{numSamples:r,seed:a,normalized:c}=s,i=c?o:si({inputs:{logits:o},backend:t,attrs:{dim:o.shape.length-1}}),l=i.shape[0],u=i.shape[1],d=new Z0(l,u,r),p=[[a]],h=t.runWebGLProgram(d,[i],"int32",p);return c||t.disposeIntermediateTensorInfo(i),h}const r$={kernelName:xl,backendName:"webgl",kernelFunc:o$};const a$=ie+`
  return -x;
`,i$=`
  vec4 result = -x;
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;function c$(n){const{inputs:e,backend:t}=n,{x:s}=e;if(t.shouldExecuteOnCPU([s])){const r=t.texData.get(s.dataId),[a,c]=Sh(r.values,s.shape,s.dtype);return t.makeTensorInfo(c,s.dtype,a)}let o;return w().getBool("WEBGL_PACK_UNARY_OPERATIONS")?o=new Re(s.shape,i$):o=new he(s.shape,a$),t.runWebGLProgram(o,[s],s.dtype)}const l$={kernelName:ro,backendName:"webgl",kernelFunc:c$};const u$=Cl;function d$(n){vt("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:e,backend:t,attrs:s}=n,{boxes:o,scores:r}=e,{maxOutputSize:a,iouThreshold:c,scoreThreshold:i}=s,l=t.readSync(o.dataId),u=t.readSync(r.dataId),{selectedIndices:d}=u$(l,u,a,c,i);return t.makeTensorInfo([d.length],"int32",new Int32Array(d))}const p$={kernelName:gl,backendName:"webgl",kernelFunc:d$};const h$=bl;function f$(n){vt("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:e,backend:t,attrs:s}=n,{boxes:o,scores:r}=e,{maxOutputSize:a,iouThreshold:c,scoreThreshold:i,padToMaxOutputSize:l}=s,u=t.readSync(o.dataId),d=t.readSync(r.dataId),{selectedIndices:p,validOutputs:h}=h$(u,d,a,c,i,l);return[t.makeTensorInfo([p.length],"int32",new Int32Array(p)),t.makeTensorInfo([],"int32",new Int32Array([h]))]}const m$={kernelName:$l,backendName:"webgl",kernelFunc:f$};const x$=wl;function g$(n){vt("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:e,backend:t,attrs:s}=n,{boxes:o,scores:r}=e,{maxOutputSize:a,iouThreshold:c,scoreThreshold:i,softNmsSigma:l}=s,u=t.readSync(o.dataId),d=t.readSync(r.dataId),p=a,h=c,f=i,g=l,{selectedIndices:x,selectedScores:m}=x$(u,d,p,h,f,g);return[t.makeTensorInfo([x.length],"int32",new Int32Array(x)),t.makeTensorInfo([m.length],"float32",new Float32Array(m))]}const C$={kernelName:vl,backendName:"webgl",kernelFunc:g$};class $${constructor(e,t,s,o){this.variableNames=["indices"],this.outputShape=[e,t],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int index = round(getIndices(coords.x));
        setOutput(mix(float(${o}), float(${s}),
                      float(index == coords.y)));
      }
    `}}const b$=n=>{const{inputs:e,backend:t,attrs:s}=n,{indices:o}=e,{dtype:r,depth:a,onValue:c,offValue:i}=s,l=E(o.shape),u=new $$(l,a,c,i),d=R({inputs:{x:o},backend:t,attrs:{shape:[l]}}),p=t.runWebGLProgram(u,[d],r);t.disposeIntermediateTensorInfo(d);const h=[...o.shape,a],f=R({inputs:{x:p},backend:t,attrs:{shape:h}});return t.disposeIntermediateTensorInfo(p),f},v$={kernelName:Il,backendName:"webgl",kernelFunc:b$};function Gt(n){const{inputs:e,backend:t}=n,{x:s}=e;if(s.dtype==="complex64"){const o=yt({inputs:{input:s},backend:t}),r=Gt({inputs:{x:o},backend:t}),a=Jt({inputs:{input:s},backend:t}),c=Gt({inputs:{x:a},backend:t}),i=Te({inputs:{real:r,imag:c},backend:t});return t.disposeIntermediateTensorInfo(o),t.disposeIntermediateTensorInfo(r),t.disposeIntermediateTensorInfo(a),t.disposeIntermediateTensorInfo(c),i}else return St({attrs:{shape:s.shape,dtype:s.dtype,value:s.dtype==="string"?"":0},backend:t})}const w$={kernelName:Rl,backendName:"webgl",kernelFunc:Gt};function oi(n){const{inputs:e,backend:t}=n,{x:s}=e;if(s.dtype==="string")throw new Error("onesLike is not supported under string dtype");if(s.dtype==="complex64"){const o=yt({inputs:{input:s},backend:t}),r=oi({inputs:{x:o},backend:t}),a=Jt({inputs:{input:s},backend:t}),c=Gt({inputs:{x:a},backend:t}),i=Te({inputs:{real:r,imag:c},backend:t});return t.disposeIntermediateTensorInfo(o),t.disposeIntermediateTensorInfo(r),t.disposeIntermediateTensorInfo(a),t.disposeIntermediateTensorInfo(c),i}else return St({attrs:{shape:s.shape,dtype:s.dtype,value:1},backend:t})}const I$={kernelName:yl,backendName:"webgl",kernelFunc:oi};function R$(n){const{inputs:e,backend:t,attrs:s}=n,{axis:o}=s;if(e.length===1)return Cn({inputs:{input:e[0]},backend:t,attrs:{dim:o}});const r=e[0].shape,a=e[0].dtype;e.forEach(u=>{Tl(r,u.shape,"All tensors passed to stack must have matching shapes"),O(a===u.dtype,()=>"All tensors passed to stack must have matching dtypes")});const c=[],i=e.map(u=>{const d=Cn({inputs:{input:u},backend:t,attrs:{dim:o}});return c.push(d),d}),l=Ga({inputs:i,backend:t,attrs:{axis:o}});return c.forEach(u=>t.disposeIntermediateTensorInfo(u)),l}const y$={kernelName:Sl,backendName:"webgl",kernelFunc:R$};class S${constructor(e,t,s){this.variableNames=["x"],this.customUniforms=[{name:"value",type:"float"}],this.outputShape=t.map((l,u)=>l[0]+e[u]+l[1]);const o=e.length,r=_(o),a=t.map(l=>l[0]).join(","),c=t.map((l,u)=>l[0]+e[u]).join(","),i=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,o);if(o===1){this.userCode=`
        int start = ${a};
        int end = ${c};

        void main() {
          int outC = getOutputCoords();
          if (outC < start || outC >= end) {
            setOutput(value);
          } else {
            setOutput(getX(outC - start));
          }
        }
      `;return}this.userCode=`
      ${r} start = ${r}(${a});
      ${r} end = ${r}(${c});

      void main() {
        ${r} outC = getOutputCoords();
        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {
          setOutput(value);
        } else {
          ${r} coords = outC - start;
          setOutput(getX(${i}));
        }
      }
    `}}class T${constructor(e,t,s){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"value",type:"float"}],this.outputShape=t.map((g,x)=>g[0]+e[x]+g[1]);const o=e.length,r=_(o),a=t.map(g=>g[0]).join(","),c=t.map((g,x)=>g[0]+e[x]).join(","),i=G("rc",o),l=G("source",o),u=`${i[o-1]} < ${this.outputShape[o-1]}`,d=o===1?"source":`vec2(${l.slice(-2).join()})`,p=[`${r} rc = outputLoc;`,`${i[o-1]} += 1;
       if(${u}) {
      `,o===1?"":`}
       rc = outputLoc;
       ${i[o-2]} += 1;
       if(${i[o-2]} < ${this.outputShape[o-2]}) {`,o===1?"":`  ${i[o-1]} += 1;
         if(${u}) {`],h=o===1?"rc < start || rc >= end":"any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))";let f="";for(let g=0,x=o===1?2:4;g<x;g++)f+=`
        ${p[g]}
        if (${h}) {
          result[${g}] = float(value);
        } else {
          ${r} source = rc - start;
          result[${g}] = getChannel(getX(${l.join()}), ${d});
        }
      `;f+=o===1?"} ":"}}",this.userCode=`
      const ${r} start = ${r}(${a});
      const ${r} end = ${r}(${c});

      void main() {
        ${r} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${f}
        setOutput(result);
      }
    `}}const ri=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{paddings:r,constantValue:a}=s;if(E(o.shape)===0){const l=r.map((u,d)=>u[0]+o.shape[d]+u[1]);return St({backend:t,attrs:{shape:l,value:a,dtype:o.dtype}})}const c=w().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new T$(o.shape,r,a):new S$(o.shape,r,a),i=[[a]];return t.runWebGLProgram(c,[o],o.dtype,i)},E$={kernelName:El,backendName:"webgl",kernelFunc:ri};const N$=`
  if(a < 0.0 && floor(b) < b){
    return NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  return (round(mod(b, 2.0)) != 1) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
`,k$=`
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  bvec4 isNaN1 = lessThan(a, vec4(0.0));
  bvec4 isNaN2 = lessThan(floor(b), b);
  bvec4 isNaN = bvec4(isNaN1.x && isNaN2.x, isNaN1.y && isNaN2.y, isNaN1.z && isNaN2.z, isNaN1.w && isNaN2.w);
  `+Be+`
  return result;
`,A$=V({opSnippet:N$,packedOpSnippet:k$}),O$={kernelName:Nl,backendName:"webgl",kernelFunc:A$};function D$(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,keepDims:a}=s,c=o.shape.length,i=[],l=X(r,o.shape);let u=l;const d=se(u,c);let p=o;d!=null&&(p=H({inputs:{x:o},backend:t,attrs:{perm:d}}),u=oe(u.length,c),i.push(p)),de("prod",u,c);let h;if(t.shouldExecuteOnCPU([p])){const f=t.texData.get(p.dataId).values,{outVals:g,outShape:x,outDtype:m}=Eh(p.shape,p.dtype,f,u);h=t.makeTensorInfo(x,m,g)}else{const[f,g]=fe(p.shape,u),x=E(g),m=R({inputs:{x:p},backend:t,attrs:{shape:[-1,x]}}),C=zn(o.dtype),$=Me(m,C,"prod",t);h=R({inputs:{x:$},backend:t,attrs:{shape:f}}),i.push(m),i.push($)}if(a){i.push(h);const f=me(h.shape,l);h=R({inputs:{x:h},backend:t,attrs:{shape:f}})}return i.forEach(f=>t.disposeIntermediateTensorInfo(f)),h}const F$={kernelName:io,backendName:"webgl",kernelFunc:D$};function P$(n){const{inputs:e,backend:t,attrs:s}=n,{paramsNestedSplits:o,paramsDenseValues:r,indices:a}=e,{outputRaggedRank:c}=s,i=o.map(m=>t.readSync(m.dataId)),l=o.map(m=>m.shape),u=t.readSync(r.dataId),d=t.readSync(a.dataId),[p,h,f]=Nh(i,l,u,r.shape,r.dtype,d,a.shape,c),g=p.map(m=>t.makeTensorInfo([m.length],"int32",m)),x=t.makeTensorInfo(f,r.dtype,h);return g.concat([x])}const _$={kernelName:kl,backendName:"webgl",kernelFunc:P$};function L$(n){const{inputs:e,backend:t}=n,{starts:s,limits:o,deltas:r}=e,a=t.readSync(s.dataId),c=t.readSync(o.dataId),i=t.readSync(r.dataId),[l,u]=kh(a,s.shape,s.dtype,c,o.shape,i,r.shape),d=t.makeTensorInfo([l.length],"int32",l),p=t.makeTensorInfo([u.length],s.dtype,u);return[d,p]}const V$={kernelName:Al,backendName:"webgl",kernelFunc:L$};function B$(n){const{inputs:e,backend:t,attrs:s}=n,{shape:o,values:r,defaultValue:a,rowPartitionTensors:c}=e,{rowPartitionTypes:i}=s,l=t.readSync(o.dataId),u=t.readSync(r.dataId),d=t.readSync(a.dataId),p=c.map(x=>t.readSync(x.dataId)),h=c.map(x=>x.shape),[f,g]=Ah(l,o.shape,u,r.shape,r.dtype,d,a.shape,p,h,i);return t.makeTensorInfo(f,r.dtype,g)}const M$={kernelName:Ol,backendName:"webgl",kernelFunc:B$};const ai=n=>{const{backend:e,attrs:t}=n,{start:s,stop:o,step:r,dtype:a}=t,c=Oh(s,o,r,a);return e.makeTensorInfo([c.length],a,c)},W$={kernelName:Dl,backendName:"webgl",kernelFunc:ai};const U$="return 1.0 / x;",G$=D({opSnippet:U$}),z$={kernelName:Fl,backendName:"webgl",kernelFunc:G$};const H$=ie+`
  return (x < 0.0) ? 0.0 : x;
`,X$=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,K$=D({opSnippet:H$,packedOpSnippet:X$}),j$={kernelName:Pl,backendName:"webgl",kernelFunc:K$};const q$=ie+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,Y$=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,Q$=D({opSnippet:q$,packedOpSnippet:Y$}),Z$={kernelName:_l,backendName:"webgl",kernelFunc:Q$};class J${constructor(e,t,s,o,r){this.variableNames=["A"],this.outputShape=[];const[a,c,i,l]=e;this.outputShape=[a,t,s,l];const u=[o&&t>1?c-1:c,o&&s>1?i-1:i],d=[o&&t>1?t-1:t,o&&s>1?s-1:s];let p;r?p="(vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC - vec2(0.5)":p="vec2(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${u[0]/d[0]},
          ${u[1]/d[1]});
      const vec2 inputShapeRC = vec2(${c}.0, ${i}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${p};

        // Compute the four integer indices.
        ivec2 sourceFloorRC = ivec2(max(sourceFracIndexRC, vec2(0.0)));
        ivec2 sourceCeilRC = ivec2(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        float topLeft = getA(b, sourceFloorRC.x, sourceFloorRC.y, d);
        float bottomLeft = getA(b, sourceCeilRC.x, sourceFloorRC.y, d);
        float topRight = getA(b, sourceFloorRC.x, sourceCeilRC.y, d);
        float bottomRight = getA(b, sourceCeilRC.x, sourceCeilRC.y, d);

        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

        float top = topLeft + (topRight - topLeft) * fracRC.y;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
        float newValue = top + (bottom - top) * fracRC.x;

        setOutput(newValue);
      }
    `}}class eb{constructor(e,t,s,o,r){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[a,c,i,l]=e;this.outputShape=[a,t,s,l];const u=[o&&t>1?c-1:c,o&&s>1?i-1:i],d=[o&&t>1?t-1:t,o&&s>1?s-1:s];let p;r?p="(vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC - vec3(0.5)":p="vec3(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${u[0]/d[0]},
          ${u[1]/d[1]},
          ${u[1]/d[1]});
      const vec3 inputShapeRC = vec3(${c}.0, ${i}.0,
                                     ${i}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${p};

        // Compute the four integer indices.
        ivec3 sourceFloorRC = ivec3(max(sourceFracIndexRC, vec3(0.0)));
        ivec3 sourceCeilRC = ivec3(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${l-1};
        bool hasNextRow = coords.z < ${s-1};

        // In parallel, construct four corners for all four components in
        // packed 2x2 cell.
        vec4 topLeft = vec4(
          getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 bottomLeft = vec4(
          getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 topRight = vec4(
          getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec4 bottomRight = vec4(
          getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec3 fracRC = sourceFracIndexRC - vec3(sourceFloorRC);

        vec4 top = mix(topLeft, topRight, fracRC.yyzz);
        vec4 bottom = mix(bottomLeft, bottomRight, fracRC.yyzz);
        vec4 newValue = mix(top, bottom, fracRC.x);

        setOutput(newValue);
      }
    `}}function tb(n){const{inputs:e,backend:t,attrs:s}=n,{images:o}=e,{alignCorners:r,halfPixelCenters:a,size:c}=s,[i,l]=c,u=w().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new eb(o.shape,i,l,r,a):new J$(o.shape,i,l,r,a);return t.runWebGLProgram(u,[o],"float32")}const nb={kernelName:Ll,backendName:"webgl",kernelFunc:tb};class sb{constructor(e,t,s){this.variableNames=["dy"],this.outputShape=[],this.outputShape=t;const[,o,r]=t,[,a,c]=e,i=[s&&a>1?o-1:o,s&&c>1?r-1:r],l=[s&&a>1?a-1:a,s&&c>1?c-1:c],u=i[0]/l[0],d=i[1]/l[1],p=1/u,h=1/d,f=Math.ceil(p)*2+2,g=Math.ceil(h)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${u});
        const float widthScale = float(${d});

        const float invHeightScale = float(${p});
        const float invWidthScale = float(${h});

        const int winHeight = int(${f});
        const int winWidth = int(${g});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(startRLerp - float(winHeight / 2));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(startCLerp - float(winWidth / 2));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${a}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${c}) {
              continue;
            }

            float dxR = float(dyR) * heightScale;
            int topDxRIndex = int(floor(dxR));
            int bottomDxRIndex = int(min(ceil(dxR), ${o-1}.0));
            float dxRLerp = dxR - float(topDxRIndex);
            float inverseDxRLerp = 1.0 - dxRLerp;

            float dxC = float(dyC) * widthScale;
            int leftDxCIndex = int(floor(dxC));
            int rightDxCIndex = int(min(ceil(dxC), ${r-1}.0));
            float dxCLerp = dxC - float(leftDxCIndex);
            float inverseDxCLerp = 1.0 - dxCLerp;

            if (r == topDxRIndex && c == leftDxCIndex) {
              // topLeft
              accumulator +=
                getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;
            }

            if (r == topDxRIndex && c == rightDxCIndex) {
              // topRight
              accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;
            }

            if (r == bottomDxRIndex && c == leftDxCIndex) {
              // bottomLeft
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;
            }

            if (r == bottomDxRIndex && c == rightDxCIndex) {
              // bottomRight
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}}function ob(n){const{inputs:e,backend:t,attrs:s}=n,{images:o,dy:r}=e,{alignCorners:a}=s,c=new sb(r.shape,o.shape,a);return t.runWebGLProgram(c,[r],r.dtype)}const rb={kernelName:Vl,backendName:"webgl",kernelFunc:ob};class ab{constructor(e,t,s,o,r){this.variableNames=["A"],this.outputShape=[];const[a,c,i,l]=e;this.outputShape=[a,t,s,l];const u=[o&&t>1?c-1:c,o&&s>1?i-1:i],d=[o&&t>1?t-1:t,o&&s>1?s-1:s],p=o?"0.5":"0.0";let h;r?h="max((vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC, vec2(0.0))":h="vec2(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${u[0]/d[0]},
          ${u[1]/d[1]});
      const vec2 inputShapeRC = vec2(${c}.0, ${i}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${h};

        // Compute the coordinators of nearest neighbor point.
        ivec2 sourceNearestRC = ivec2(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${p})));
        float newValue = getA(b, sourceNearestRC.x, sourceNearestRC.y, d);

        setOutput(newValue);
      }
    `}}class ib{constructor(e,t,s,o,r){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[a,c,i,l]=e;this.outputShape=[a,t,s,l];const u=[o&&t>1?c-1:c,o&&s>1?i-1:i],d=[o&&t>1?t-1:t,o&&s>1?s-1:s],p=o?"0.5":"0.0";let h;r?h="max((vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC, vec3(0.0))":h="vec3(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${u[0]/d[0]},
          ${u[1]/d[1]},
          ${u[1]/d[1]});
      const vec3 inputShapeRC = vec3(${c}.0, ${i}.0,
                                     ${i}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${h};

        // Compute the coordinators of nearest neighbor point.
        ivec3 sourceNearestRC = ivec3(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${p})));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${l-1};
        bool hasNextRow = coords.z < ${s-1};

        vec4 newValue = vec4(
          getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d),
          hasNextCol ? getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d + 1) : 0.0);

        setOutput(newValue);
      }
    `}}function cb(n){const{inputs:e,backend:t,attrs:s}=n,{images:o}=e,{alignCorners:r,halfPixelCenters:a,size:c}=s,[i,l]=c,u=w().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new ib(o.shape,i,l,r,a):new ab(o.shape,i,l,r,a);return t.runWebGLProgram(u,[o],o.dtype)}const lb={kernelName:Bl,backendName:"webgl",kernelFunc:cb};class ub{constructor(e,t,s){this.variableNames=["dy"],this.outputShape=[],this.outputShape=t;const[,o,r]=t,[,a,c]=e,i=[s&&a>1?o-1:o,s&&c>1?r-1:r],l=[s&&a>1?a-1:a,s&&c>1?c-1:c],u=i[0]/l[0],d=i[1]/l[1],p=1/u,h=1/d,f=Math.ceil(p)*2+2,g=Math.ceil(h)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${u});
        const float widthScale = float(${d});

        const float invHeightScale = float(${p});
        const float invWidthScale = float(${h});

        const int winHeight = int(${f});
        const int winWidth = int(${g});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(floor(startRLerp - float(winHeight / 2)));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(floor(startCLerp - float(winWidth / 2)));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${a}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${c}) {
              continue;
            }

            float sourceFracRow =
              float(${i[0]}) *
                (float(dyR) / float(${l[0]}));

            float sourceFracCol =
                float(${i[1]}) *
                  (float(dyC) / float(${l[1]}));

            int sourceNearestRow = int(min(
                float(int(${o}) - 1),
                ${s} ? float(round(sourceFracRow)) :
                                  float(floor(sourceFracRow))));

            int sourceNearestCol = int(min(
                float(int(${r}) - 1),
                ${s} ? float(round(sourceFracCol)) :
                                  float(floor(sourceFracCol))));

            if (r == sourceNearestRow && c == sourceNearestCol) {
              accumulator += getDy(b, dyR, dyC, d);
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}}function db(n){const{inputs:e,backend:t,attrs:s}=n,{images:o,dy:r}=e,{alignCorners:a}=s,c=new ub(r.shape,o.shape,a);return t.runWebGLProgram(c,[r],r.dtype)}const pb={kernelName:Ml,backendName:"webgl",kernelFunc:db};class hb{constructor(e,t){this.variableNames=["x"];const s=e.length;if(s>4)throw new Error(`WebGL backend: Reverse of rank-${s} tensor is not yet supported`);if(this.outputShape=e,s===1){this.userCode=`
        void main() {
          int coord = getOutputCoords();
          setOutput(getX(${e[0]} - coord - 1));
        }
      `;return}const o=c=>t.indexOf(c)!==-1&&e[c]!==1?`${e[c]} - coords[${c}] - 1`:`coords[${c}]`,r=e.map((c,i)=>o(i)).join(","),a=_(s);this.userCode=`
      void main() {
        ${a} coords = getOutputCoords();
        setOutput(getX(${r}));
      }
    `}}class fb{constructor(e,t){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0;const s=e.length;if(s>4)throw new Error(`WebGL backend: Reverse of rank-${s} tensor is not yet supported`);this.outputShape=e;const o=G("rc",s),r=`${o[s-1]} + 1 < ${this.outputShape[s-1]}`,a=`${o[s-2]} + 1 < ${this.outputShape[s-2]}`,c=_(s);s===1?this.userCode=`
        void main(){
          int rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = getChannel(getX(${e[0]} - rc - 1),
            ${e[0]} - rc - 1);
          if(${r}){
              result.g = getChannel(getX(${e[0]} - (rc  + 1) - 1),
                ${e[0]} - (rc  + 1) - 1);
          }
          setOutput(result);
        }
      `:this.userCode=`
        void main() {
          ${c} rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = ${i(o.slice())};
          if(${r}){
            result.g = ${l(o.slice())};
          }
          if(${a}) {
            result.b = ${u(o.slice())};
            if(${r}) {
              result.a = ${d(o.slice())};
            }
          }
          setOutput(result);
        }
    `;function i(f){return p(f)}function l(f){return f[s-1]="("+f[s-1]+" + 1)",p(f)}function u(f){return f[s-2]="("+f[s-2]+" + 1)",p(f)}function d(f){return f[s-1]="("+f[s-1]+" + 1)",f[s-2]="("+f[s-2]+" + 1)",p(f)}function p(f){const g=e.map((C,$)=>h($,f)),x=g.join(","),m=g.slice(-2).join(",");return`getChannel(getX(${x}), vec2(${m}))`}function h(f,g){return t.indexOf(f)!==-1&&e[f]!==1?`${e[f]} - ${g[f]} - 1`:`${g[f]}`}}}function mb(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{dims:r}=s,a=o.shape.length,c=X(r,o.shape);if(a===0)return Z({inputs:{x:o},backend:t});const i=w().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new fb(o.shape,c):new hb(o.shape,c);return t.runWebGLProgram(i,[o],o.dtype)}const xb={kernelName:Wl,backendName:"webgl",kernelFunc:mb};class gb{constructor(e,t){this.variableNames=["Image"],this.outputShape=[],this.customUniforms=[{name:"params",type:"vec4"}];const s=e[1],o=e[2];this.outputShape=e;let r="";typeof t=="number"?r=`float outputValue = ${t.toFixed(2)};`:r=`
        vec3 fill = vec3(${t.join(",")});
        float outputValue = fill[coords[3]];`,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];
          int y = coords[1];
          float coordXFloat = (float(x) - params[0]) * params[3] -
            (float(y) - params[1]) * params[2];
          float coordYFloat = (float(x) - params[0]) * params[2] +
            (float(y) - params[1]) * params[3];
          int coordX = int(round(coordXFloat + params[0]));
          int coordY = int(round(coordYFloat + params[1]));
          ${r}
          if(coordX >= 0 && coordX < ${o} && coordY >= 0 && coordY < ${s}) {
            outputValue = getImage(coords[0], coordY, coordX, coords[3]);
          }
          setOutput(outputValue);
        }
    `}}const Cb={kernelName:Ul,backendName:"webgl",kernelFunc:({inputs:n,attrs:e,backend:t})=>{const{image:s}=n,{radians:o,fillValue:r,center:a}=e,c=t,i=new gb(s.shape,r),[l,u]=To(a,s.shape[1],s.shape[2]),d=[[l,u,Math.sin(o),Math.cos(o)]];return c.runWebGLProgram(i,[s],s.dtype,d)}};const $b=`
  // OpenGL ES does not support round function.
  // The algorithm is based on banker's rounding.
  float base = floor(x);
  if ((x - base) < 0.5) {
    return floor(x);
  } else if ((x - base) > 0.5) {
    return ceil(x);
  } else {
    if (mod(base, 2.0) == 0.0) {
      return base;
    } else {
      return base + 1.0;
    }
  }
`,bb=D({opSnippet:$b}),vb={kernelName:Gl,backendName:"webgl",kernelFunc:bb};const wb="return inversesqrt(x);",Ib=D({opSnippet:wb,cpuKernelImpl:Dh}),Rb={kernelName:Ln,backendName:"webgl",kernelFunc:Ib};class xs{constructor(e,t,s,o,r,a,c=!0,i=!1){this.variableNames=["updates","indices","defaultValue"],this.outputShape=a;const l=_(r.length),u=_(a.length);let d="";s===1?d="i":s===2&&(d="i, j");const p=`getIndices(${d})`;let h="";o===1?h="i":o===2&&(h="i, coords[1]");const f=`getUpdates(${h})`;let g="";i&&(g="coords[0], coords[1]");const x=`getDefaultValue(${g})`,m=t>1?"strides[j]":"strides";this.userCode=`
        ${l} strides = ${l}(${r});

        void main() {
          ${u} coords = getOutputCoords();
          float sum = 0.0;
          bool found = false;
          for (int i = 0; i < ${e}; i++) {
            int flattenedIndex = 0;
            for (int j = 0; j < ${t}; j++) {
              int index = round(${p});
              flattenedIndex += index * ${m};
            }
            if (flattenedIndex == coords[0]) {
              sum += ${f};
              found = true;
            }
          }
          setOutput(mix(${x}, sum, float(found)));
        }
      `}}class yb{constructor(e,t,s,o,r,a,c=!0,i=!1){this.variableNames=["updates","indices","defaultValue"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=a;const l=_(r.length),u=_(a.length);let d="";s===1?d="i":s===2&&(d="i, j");const p=`getIndices(${d})`;let h="";o===1?h="i":o===2&&(h="i, coords[1]");const f=`getUpdates(${h})`;let g="";i&&(g="coords[0], coords[1]");const x=`getDefaultValue(${g})`,m=t>1?"strides[j]":"strides",C=t>1?"strides[j + 1]":"strides";this.userCode=`
        ${l} strides = ${l}(${r});

        void main() {
          ${u} coords = getOutputCoords();
          vec4 sum = vec4(0.);
          vec4 found = vec4(0.);
          for (int i = 0; i < ${e}; i+=2) {
            ivec2 flattenedIndex = ivec2(0);
            for (int j = 0; j < ${t}; j+=2) {
              ivec4 index = round(${p});
              flattenedIndex += index.xz * ${m};
              if (j + 1 < ${t}) {
                flattenedIndex += index.yw * ${C};
              }
            }
            if (flattenedIndex[0] == coords[0] || flattenedIndex[1] == coords[0] ||
                flattenedIndex[0] == coords[0] + 1 || flattenedIndex[1] == coords[0] + 1) {
              vec4 updVals = ${f};
              if (flattenedIndex[0] == coords[0]) {
                sum.xy += updVals.xy;
                found.xy = vec2(1.);
              } else if (flattenedIndex[0] == coords[0] + 1) {
                sum.zw += updVals.xy;
                found.zw = vec2(1.);
              }
              if (flattenedIndex[1] == coords[0]) {
                sum.xy += updVals.zw;
                found.xy = vec2(1.);
              } else if (flattenedIndex[1] == coords[0] + 1) {
                sum.zw += updVals.zw;
                found.zw = vec2(1.);
              }
            }
          }
          setOutput(mix(${x}, sum, found));
        }
      `}}function Sb(n){const{inputs:e,backend:t,attrs:s}=n,{indices:o,updates:r}=e,{shape:a}=s,{sliceRank:c,numUpdates:i,sliceSize:l,strides:u,outputSize:d}=Ht(r,o,a),p=[d/l,l];if(d===0)return t.makeTensorInfo(a,o.dtype);const h=R({inputs:{x:o},backend:t,attrs:{shape:[i,c]}}),f=R({inputs:{x:r},backend:t,attrs:{shape:[i,l]}}),g=t.makeTensorInfo([],"float32",new Float32Array([0]));let x;w().getBool("WEBGL_PACK")?x=new yb(i,c,h.shape.length,f.shape.length,u,p):x=new xs(i,c,h.shape.length,f.shape.length,u,p);const m=t.runWebGLProgram(x,[f,h,g],f.dtype),C=R({inputs:{x:m},backend:t,attrs:{shape:a}});return t.disposeIntermediateTensorInfo(h),t.disposeIntermediateTensorInfo(f),t.disposeIntermediateTensorInfo(m),t.disposeIntermediateTensorInfo(g),C}const Tb={kernelName:zl,backendName:"webgl",kernelFunc:Sb};class Eb{constructor(e,t,s,o){this.variableNames=["sortedSequence","values"],this.customUniforms=[{name:"numInputs",type:"int"}],this.outputShape=[e,s];const r="while (left < right) {",a=`for (int i = 0; i < ${Math.ceil(Math.log2(t+1))}; ++i) { if (left >= right) break;`,c=w().getNumber("WEBGL_VERSION")===2?r:a,i=o==="left"?"<":"<=";this.userCode=`
       int findBound(int batch, float value) {
         int left = 0;
         int right = numInputs;
         int mid;
         ${c}
           mid = (left + right) / 2;
           if (getSortedSequence(batch, mid) ${i} value) {
             left = mid + 1;
           } else {
             right = mid;
           }
         }
         return right;
       }

       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int valueIndex = coords[1];

         float value = getValues(batch, valueIndex);

         setOutput(float(findBound(batch, value)));
       }
     `}}function Nb(n){const{inputs:e,backend:t,attrs:s}=n,{sortedSequence:o,values:r}=e,{side:a}=s,c=new Eb(o.shape[0],o.shape[1],r.shape[1],a),i=[[o.shape[1]]];return t.runWebGLProgram(c,[o,r],"int32",i)}const kb={kernelName:Hl,backendName:"webgl",kernelFunc:Nb};class Ab{constructor(e,t,s){this.variableNames=["c","a","b"],this.outputShape=t;let o,r;if(s>4)throw Error(`Where for rank ${s} is not yet supported`);if(s===1)r="resRC",o="resRC";else{const c=["resRC.x","resRC.y","resRC.z","resRC.w"],i=[],l=[];for(let u=0;u<t.length;u++)l.push(`${c[u]}`),u<e&&i.push(`${c[u]}`);o=i.join(),r=l.join()}const a=_(s);this.userCode=`
      void main() {
        ${a} resRC = getOutputCoords();
        float cVal = getC(${o});
        if (cVal >= 1.0) {
          setOutput(getA(${r}));
        } else {
          setOutput(getB(${r}));
        }
      }
    `}}function Ob(n){const{inputs:e,backend:t}=n,{condition:s,t:o,e:r}=e,a=new Ab(s.shape.length,o.shape,o.shape.length);return t.runWebGLProgram(a,[s,o,r],ye(o.dtype,r.dtype))}const Db={kernelName:Xl,backendName:"webgl",kernelFunc:Ob};const Fb=`
  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
  // see: https://arxiv.org/abs/1706.02515
  float scaleAlpha = ${ko};
  float scale = ${Ao};
  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);
`,Pb=D({opSnippet:Fb}),_b={kernelName:Kl,backendName:"webgl",kernelFunc:Pb};const Lb=nt+`
  return 1.0 / (1.0 + exp(-1.0 * x));
`,Vb=`
  vec4 result = 1.0 / (1.0 + exp(-1.0 * x));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,Bb=D({opSnippet:Lb,packedOpSnippet:Vb,cpuKernelImpl:Ph}),Mb={kernelName:Vn,backendName:"webgl",kernelFunc:Bb};const Wb=`
  if (isnan(x)) { return 0.0; }
  return sign(x);
`,Ub=D({opSnippet:Wb}),Gb={kernelName:jl,backendName:"webgl",kernelFunc:Ub};const zb=nt+`
  return sin(x);
`,Hb=`
  vec4 result = sin(x);
  bvec4 isNaN = isnan(x);
  ${Be}
  return result;
`,Xb=D({opSnippet:zb,packedOpSnippet:Hb}),Kb={kernelName:ql,backendName:"webgl",kernelFunc:Xb};const jb=`
  float e2x = exp(x);
  return (e2x - 1.0 / e2x) / 2.0;
`,qb=D({opSnippet:jb}),Yb={kernelName:Yl,backendName:"webgl",kernelFunc:qb};const Qb=`
  float epsilon = 1.1920928955078125e-7;
  float threshold = log(epsilon) + 2.0;

  bool too_large = x > -threshold;
  bool too_small = x < threshold;

  float result;
  float exp_x = exp(x);

  if (too_large){
    result = x;
  }
  else if (too_small){
    result = exp_x;
  }
  else{
    result = log(exp_x + 1.0);
  }
  return result;
`,Zb=D({opSnippet:Qb}),Jb={kernelName:Ql,backendName:"webgl",kernelFunc:Zb};const e1=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{blockShape:r,paddings:a}=s;O(o.shape.length<=4,()=>"spaceToBatchND for rank > 4 with a WebGL backend not implemented yet");const c=r.reduce((m,C)=>m*C),i=[[0,0]];i.push(...a);for(let m=1+r.length;m<o.shape.length;++m)i.push([0,0]);const l=[],u=ri({inputs:{x:o},backend:t,attrs:{paddings:i,constantValue:0}}),d=Qn(u.shape,r,c,!1),p=Zn(d.length,r.length,!1),h=Jn(u.shape,r,c,!1),f=R({inputs:{x:u},backend:t,attrs:{shape:d}}),g=H({inputs:{x:f},backend:t,attrs:{perm:p}}),x=R({inputs:{x:g},backend:t,attrs:{shape:h}});return l.push(u),l.push(f),l.push(g),l.forEach(m=>t.disposeIntermediateTensorInfo(m)),x},t1={kernelName:Zl,backendName:"webgl",kernelFunc:e1};function n1(n){const{inputs:e,backend:t}=n,{indices:s,values:o,denseShape:r,defaultValue:a}=e;if(r.shape.length!==1)throw new Error(`Dense shape must be a vector, saw:
         ${r.shape}`);if(s.shape.length!==2)throw new Error(`Indices must be a matrix, saw:
         ${s.shape}`);if(o.shape.length!==1)throw new Error(`Values must be a vector, saw:
         ${o.shape}`);if(a.shape.length!==0)throw new Error(`Default value must be a scalar, saw:
        ${a.shape}`);const c=t.readSync(s.dataId),i=t.readSync(o.dataId),l=t.readSync(r.dataId),u=t.readSync(a.dataId)[0],[d,p,h,f,g]=Lh(c,s.shape,s.dtype,i,o.dtype,l,u);return[t.makeTensorInfo(p,s.dtype,d),t.makeTensorInfo([p[0]],o.dtype,h),t.makeTensorInfo([f.length],"bool",new Uint8Array(f.map(x=>Number(x)))),t.makeTensorInfo([g.length],s.dtype,new Int32Array(g))]}const s1={kernelName:Jl,backendName:"webgl",kernelFunc:n1};function o1(n){const{inputs:e,backend:t}=n,{inputIndices:s,inputShape:o,newShape:r}=e;if(s.shape.length!==2)throw new Error(`Input indices should be a matrix but received shape ${s.shape}`);if(o.shape.length!==1)throw new Error(`Input shape should be a vector but received shape ${o.shape}`);if(r.shape.length!==1)throw new Error(`Target shape should be a vector but received shape ${r.shape}`);const a=Array.from(t.readSync(o.dataId)),c=t.readSync(s.dataId),i=Array.from(t.readSync(r.dataId)),[l,u,d]=Vh(c,s.shape,s.dtype,a,i);return[t.makeTensorInfo(u,s.dtype,l),t.makeTensorInfo([d.length],r.dtype,new Int32Array(d))]}const r1={kernelName:eu,backendName:"webgl",kernelFunc:o1};function a1(n){const{inputs:e,backend:t}=n,{data:s,indices:o,segmentIds:r}=e;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.shape.length!==1)throw new Error(`Indices should be a vector but received shape
              ${o.shape}`);if(r.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
              ${r.shape}`);const a=t.readSync(s.dataId),c=t.readSync(o.dataId),i=t.readSync(r.dataId),[l,u]=Na(a,s.shape,s.dtype,c,i,!0);return t.makeTensorInfo(u,s.dtype,l)}const i1={kernelName:tu,backendName:"webgl",kernelFunc:a1};function c1(n){const{inputs:e,backend:t}=n,{data:s,indices:o,segmentIds:r}=e;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.shape.length!==1)throw new Error(`Indices should be a vector but received shape
             ${o.shape}`);if(r.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
             ${r.shape}`);const a=t.readSync(s.dataId),c=t.readSync(o.dataId),i=t.readSync(r.dataId),[l,u]=Na(a,s.shape,s.dtype,c,i);return t.makeTensorInfo(u,s.dtype,l)}const l1={kernelName:nu,backendName:"webgl",kernelFunc:c1};function u1(n){const{inputs:e,backend:t,attrs:s}=n,{sparseIndices:o,sparseValues:r,defaultValue:a}=e,{outputShape:c}=s,{sliceRank:i,numUpdates:l,sliceSize:u,strides:d,outputSize:p}=Ht(r,o,c),h=!1;if(r.dtype==="string"){const m=t.bufferSync(o),C=t.bufferSync(r),$=zt(t.readSync(a.dataId)[0]),b=Fh(m,C,c,p,u,l,i,d,$,h);return t.makeTensorInfo(c,b.dtype,b.values)}const f=new xs(l,i,o.shape.length,r.shape.length,d,[p,1],h),g=t.runWebGLProgram(f,[r,o,a],r.dtype),x=R({inputs:{x:g},backend:t,attrs:{shape:c}});return t.disposeIntermediateTensorInfo(g),x}const d1={kernelName:su,backendName:"webgl",kernelFunc:u1};function p1(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{numOrSizeSplits:r,axis:a}=s,c=X(a,o.shape)[0],i=Go(o,r,c),l=o.shape.length,u=new Array(l).fill(0),d=o.shape.slice();return i.map(p=>{const h=[...d];h[c]=p;const f=st({inputs:{x:o},backend:t,attrs:{begin:u,size:h}});return u[c]+=p,f})}const h1={kernelName:ou,backendName:"webgl",kernelFunc:p1};const qs="return sqrt(x);",f1=D({opSnippet:qs,packedOpSnippet:qs,cpuKernelImpl:Bh}),m1={kernelName:Bn,backendName:"webgl",kernelFunc:f1};const x1="return x * x;",g1=D({opSnippet:x1}),C1={kernelName:ru,backendName:"webgl",kernelFunc:g1};const Ys="return (a - b) * (a - b);",$1=V({opSnippet:Ys,packedOpSnippet:Ys}),b1={kernelName:Mn,backendName:"webgl",kernelFunc:$1};function v1(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e;if(o.dtype!=="string")throw new Error("Input must be of datatype string");const r=t.readSync(o.dataId),a=Ce(r),c=Mh(a,"string",s);return t.makeTensorInfo(o.shape,"string",c)}const w1={kernelName:Wn,backendName:"webgl",kernelFunc:v1};function I1({inputs:n,attrs:e,backend:t}){const{x:s}=n,o=ie+`
    return x > 0.0 ? 1.0 : float(${e.alpha});
  `,r=new he(s.shape,o);return t.runWebGLProgram(r,[s],s.dtype)}const R1={kernelName:au,backendName:"webgl",kernelFunc:I1};class y1{constructor(e,t,s){this.variableNames=["x"],this.outputShape=s;const o=s.length,r=_(s.length),a=_(s.length);let c="";if(o===1)c="coords * strides + begin";else{let i=0;c=s.map((l,u)=>(i++,s.length===1?`coords * strides[${u}] + begin[${u}]`:`coords[${i-1}] * strides[${u}] + begin[${u}]`)).join(",")}this.userCode=`
      ${r} begin = ${r}(${e});
      ${r} strides = ${r}(${t});

      void main() {
        ${a} coords = getOutputCoords();
        setOutput(getX(${c}));
      }
    `}}function S1(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{begin:r,end:a,strides:c,beginMask:i,endMask:l,ellipsisMask:u,newAxisMask:d,shrinkAxisMask:p}=s,{finalShapeSparse:h,finalShape:f,isIdentity:g,sliceDim0:x,isSimpleSlice:m,begin:C,end:$,strides:b}=vo(o.shape,r,a,c,i,l,u,d,p);let v;if(g)v=R({inputs:{x:o},backend:t,attrs:{shape:f}});else if(x||m){O(o.shape.length>=1,()=>`Input must have rank at least 1, got: ${o.shape.length}`);const S=po(C,$,b),I=st({inputs:{x:o},backend:t,attrs:{begin:C,size:S}});v=R({inputs:{x:I},backend:t,attrs:{shape:f}}),t.disposeIntermediateTensorInfo(I)}else if(t.shouldExecuteOnCPU([o])){const I=t.readSync(o.dataId),A=j(o.shape,o.dtype,I),k=Wh(h,A,b,C);v=t.makeTensorInfo(f,o.dtype,k.values)}else{const I=new y1(C,b,h);v=t.runWebGLProgram(I,[o],o.dtype)}const T=R({inputs:{x:v},backend:t,attrs:{shape:f}});return t.disposeIntermediateTensorInfo(v),T}const T1={kernelName:iu,backendName:"webgl",kernelFunc:S1};function E1(n){const{inputs:e,backend:t,attrs:s}=n,{separator:o,nGramWidths:r,leftPad:a,rightPad:c,padWidth:i,preserveShortSequences:l}=s,{data:u,dataSplits:d}=e,p=t.readSync(u.dataId),h=t.readSync(d.dataId),[f,g]=Uh(p,h,o,r,a,c,i,l);return[t.makeTensorInfo([f.length],"string",f),t.makeTensorInfo(d.shape,"int32",g)]}const N1={kernelName:cu,backendName:"webgl",kernelFunc:E1};function k1(n){const{inputs:e,backend:t,attrs:s}=n,{skipEmpty:o}=s,{input:r,delimiter:a}=e;if(r.dtype!=="string")throw new Error("Input must be of datatype string");if(r.shape.length!==1)throw new Error(`Input must be a vector, got shape: ${r.shape}`);if(a.shape.length!==0)throw new Error(`Delimiter must be a scalar, got shape: ${a.shape}`);const c=t.readSync(r.dataId),i=t.readSync(a.dataId)[0],[l,u,d]=Gh(c,i,o),p=u.length;return[t.makeTensorInfo([p,2],"int32",l),t.makeTensorInfo([p],"string",u),t.makeTensorInfo([2],"int32",new Int32Array(d))]}const A1={kernelName:lu,backendName:"webgl",kernelFunc:k1};function O1(n){const{inputs:e,backend:t,attrs:s}=n,{numBuckets:o}=s,{input:r}=e;if(r.dtype!=="string")throw new Error("Input must be of datatype string");if(o<=0)throw new Error("Number of buckets must be at least 1");const a=t.readSync(r.dataId),c=zh(a,o);return t.makeTensorInfo(r.shape,"int32",c)}const D1={kernelName:uu,backendName:"webgl",kernelFunc:O1};const F1="return tan(x);",P1=D({opSnippet:F1}),_1={kernelName:du,backendName:"webgl",kernelFunc:P1};const L1=`
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`,V1=D({opSnippet:L1}),B1={kernelName:pu,backendName:"webgl",kernelFunc:V1};function M1(n){const{inputs:e,backend:t,attrs:s}=n,{tensor:o,indices:r,updates:a}=e,{sliceRank:c,numUpdates:i,sliceSize:l,strides:u,outputSize:d}=Ht(a,r,o.shape),p=[d/l,l];if(d===0)return t.makeTensorInfo(o.shape,r.dtype);const h=R({inputs:{x:r},backend:t,attrs:{shape:[i,c]}}),f=R({inputs:{x:a},backend:t,attrs:{shape:[i,l]}}),g=R({inputs:{x:o},backend:t,attrs:{shape:p}}),x=new xs(i,c,h.shape.length,f.shape.length,u,p,!1,!0),m=t.runWebGLProgram(x,[f,h,g],g.dtype),C=R({inputs:{x:m},backend:t,attrs:{shape:o.shape}});return t.disposeIntermediateTensorInfo(h),t.disposeIntermediateTensorInfo(f),t.disposeIntermediateTensorInfo(g),t.disposeIntermediateTensorInfo(m),C}const W1={kernelName:hu,backendName:"webgl",kernelFunc:M1};class U1{constructor(e,t){this.variableNames=["A"];const s=new Array(e.length);for(let a=0;a<s.length;a++)s[a]=e[a]*t[a];this.outputShape=s,this.rank=s.length;const o=_(this.rank),r=G1(e);this.userCode=`
      void main() {
        ${o} resRC = getOutputCoords();
        setOutput(getA(${r}));
      }
    `}}function G1(n){const e=n.length;if(e>5)throw Error(`Tile for rank ${e} is not yet supported`);if(e===1)return`imod(resRC, ${n[0]})`;const t=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u"],s=[];for(let o=0;o<n.length;o++)s.push(`imod(${t[o]}, ${n[o]})`);return s.join()}function ii(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{reps:r}=s;if(o.dtype==="string"||o.shape.length>5){const i=t.readSync(o.dataId),l=o.dtype==="string"?i.map(p=>zt(p)):i,u=j(o.shape,o.dtype,l),d=Xh(u,r);return t.makeTensorInfo(d.shape,d.dtype,d.values)}const a=new U1(o.shape,r);return t.runWebGLProgram(a,[o],o.dtype)}const z1={kernelName:fu,backendName:"webgl",kernelFunc:ii};class H1{constructor(e){this.variableNames=["x","indices"],this.customUniforms=[{name:"n",type:"int"},{name:"firstPass",type:"int"},{name:"negativeInf",type:"float"},{name:"dir",type:"int"},{name:"inc",type:"int"}],this.outputShape=e,this.userCode=`
       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // We compare elements pair-wise within a group of size 2 * inc.
         // The comparing rule for each group alternates between ascending
         // and descending. Within each group, we compare each pair at
         // positions i and i+inc. To decide whether an element at position i
         // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
         // inc, it is in the first half of the group, we denote it as x0,
         // otherwise we denote it as x1.
         // For example, as shown in the Bitonic top K paper referenced above,
         // Figure5(a) shows that element[1] is in the
         // second half of the group when group size is 2, but it is in the
         // first half of the group when group size is 4.

         bool isFirstInPair = imod(elemIdx, 2 * inc) < inc;
         int i = isFirstInPair ? elemIdx : elemIdx - inc;

         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + inc : int(getIndices(batch, i + inc));
         float x0 = i0 < n ? getX(batch, i0) : negativeInf;
         float x1 = i1 < n ? getX(batch, i1) : negativeInf;

         // Denotes which direction indices are in (ascending or descending).
         bool reverse = imod(elemIdx, 2 * dir) >= dir;
         bool isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
         if (reverse == isGreater) { // Elements in opposite order of direction
           int iTemp = i0;
           i0 = i1;
           i1 = iTemp;
         }
         if (isFirstInPair) {
            setOutput(float(i0));
         } else {
            setOutput(float(i1));
         }
       }
     `}}class X1{constructor(e){this.variableNames=["x","indices"],this.customUniforms=[{name:"n",type:"int"},{name:"firstPass",type:"int"},{name:"k",type:"int"}],this.outputShape=e,this.userCode=`
    void main() {
         // Takes max of indices (0, k), (1, k + 1), (2, k + 2) ...
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // The output size is half of the previous size.
         // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _ (k=4),
         // we only need to output the indices at positions |, the indices at
         // positions _ can be thrown away, see Figure5(b) After Phase 2
         // (Merge phase) in the Bitonic Top K paper referenced above.
         // For example, the paper shows we only need to output the orange bars.
         // The output sequence should look like this | | | | | | | |.
         // Because the sequence is halved, to map the output index back
         // to the previous sequence to find the corresponding value,
         // we need to double the index. When we double the index,
         // we basically interpolate a position, so 2i looks like
         // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k position
         // of each 2k positions by - elemIdx % k. E.g. for output at
         // index 4,5,6,7, we want to get the corresponding element at
         // original index 8,9,10,11, for output at index 8,9,10,11,
         // we want to get the corresponding element at original index
         // 16,17,18,19, so on and so forth.

         int i = elemIdx < k ? elemIdx : (elemIdx * 2 - imod(elemIdx, k));
         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + k : int(getIndices(batch, i + k));

         float x0 = getX(batch, i0);
         float x1 = i1 < n ? getX(batch, i1) : x0;

         setOutput(x0 >= x1 ? float(i0) : float(i1));
       }
     `}}function Ne(n,e){e!==null&&n.disposeIntermediateTensorInfo(e)}function Qs(n){let e=1;for(;e<n;)e*=2;return e}function K1(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{k:r,sorted:a}=s,c=w().getNumber("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD"),i=w().getNumber("TOPK_K_CPU_HANDOFF_THRESHOLD"),l=o.shape,u=l[l.length-1];if(t.shouldExecuteOnCPU([o])||u<c||r>i){const k=t.readSync(o.dataId),[F,P]=Kh(k,l,o.dtype,r,a);return[t.makeTensorInfo(F.shape,F.dtype,F.values),t.makeTensorInfo(P.shape,P.dtype,P.values)]}if(r===0)return l[l.length-1]=0,[t.makeTensorInfo(l,o.dtype,[]),t.makeTensorInfo(l,"int32",[])];if(u===1)return[o,St({attrs:{shape:l,dtype:"int32",value:0},backend:t})];const d=t.texData.get(o.dataId),p=d!==null&&d.isPacked,h=p?t.unpackTensor(o):o,g=E(l)/u,x=R({inputs:{x:h},attrs:{shape:[g,u]},backend:t});p&&Ne(t,h);const m=Qs(r),C=Qs(u);let $=null;const b=()=>$===null?[x,x]:[x,$],v=(k,F,P)=>{const pe=b(),Q=new H1(P),re=[[u],[$===null?1:0],[Number.NEGATIVE_INFINITY],[k],[F]],ce=$;$=t.runWebGLProgram(Q,pe,"int32",re),Ne(t,ce)};for(let k=1;k<m;k*=2){const F=k*2;for(let P=k;P>=1;P/=2)v(F,P,[g,C])}for(let k=C;k>m;k/=2){const F=b(),P=new X1([g,k/2]),Q=[[u],[$===null?1:0],[m]],ee=$;$=t.runWebGLProgram(P,F,"int32",Q),Ne(t,ee);const re=m/2,ce=re*2;for(let U=re;U>=1;U/=2)v(ce,U,$.shape)}let T=$;$=st({inputs:{x:$},backend:t,attrs:{begin:0,size:[g,r]}}),Ne(t,T);let S=Ja({inputs:{x,indices:$},backend:t,attrs:{axis:1,batchDims:1}});Ne(t,x);const I=l.slice(0,-1);I.push(r),T=$,$=R({inputs:{x:$},attrs:{shape:I},backend:t}),Ne(t,T);const A=S;return S=R({inputs:{x:S},attrs:{shape:I},backend:t}),Ne(t,A),[S,$]}const j1={kernelName:mu,backendName:"webgl",kernelFunc:K1};class q1{constructor(e,t,s,o,r,a){this.variableNames=["Image","Transforms"],this.outputShape=a;const c=s==="nearest"?1:2;let i;switch(o){case"constant":i=1;break;case"reflect":i=2;break;case"wrap":i=3;break;case"nearest":i=4;break;default:i=1;break}this.userCode=`
            float mapCoord(float outCoord, float len) {
              float inCoord = outCoord;
              if(${i} == 2) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    if (inCoord < sz2) {
                      inCoord = sz2 * float(int(float(-inCoord / sz2))) +
                      inCoord;
                    }
                    inCoord = inCoord < -len ? inCoord + sz2 : -inCoord - 1.0;
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    inCoord -= sz2 * float(int(float(inCoord / sz2)));
                    if (inCoord >= len) {
                      inCoord = sz2 - inCoord - 1.0;
                    }
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${i} == 3) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord += len * (float(int(float(-inCoord / sz))) + 1.0);
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord -= len * float(int(float(inCoord / sz)));
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${i} == 4) {
                return clamp(outCoord, 0.0, len - 1.0);
              } else {
                return outCoord;
              }
            }

            float readWithFillValue(int batch, int coordY, int coordX,
              int channel) {
              float outputValue;
              if (0 <= coordY && coordY < ${e} && 0 <= coordX && coordX < ${t}) {
                  outputValue = getImage(batch, coordY, coordX, channel);
              } else {
                outputValue = float(${r});
              }
              return outputValue;
            }

            void main() {
              ivec4 coords = getOutputCoords();
              float outputValue;
              int batch = coords[0];
              int x = coords[2];
              int y = coords[1];
              int channel = coords[3];
              float xf = float(x);
              float yf = float(y);
              float a1 = getTransforms(batch, 0);
              float a2 = getTransforms(batch, 1);
              float a3 = getTransforms(batch, 2);
              float b1 = getTransforms(batch, 3);
              float b2 = getTransforms(batch, 4);
              float b3 = getTransforms(batch, 5);
              float c1 = getTransforms(batch, 6);
              float c2 = getTransforms(batch, 7);
              float projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = float(${r});
              } else {
                float inX = (a1 * xf + a2 * yf + a3) / projection;
                float inY = (b1 * xf + b2 * yf + b3) / projection;
                float mapX = mapCoord(inX, float(${t}));
                float mapY = mapCoord(inY, float(${e}));

                if (${c} == 1) {
                  int coordY = int(round(mapY));
                  int coordX = int(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  float yFloor = floor(mapY);
                  float xFloor = floor(mapX);
                  float yCeil = yFloor + 1.0;
                  float xCeil = xFloor + 1.0;
                  float valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, int(yFloor), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yFloor), int(xCeil), channel);
                  float valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, int(yCeil), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yCeil), int(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutput(outputValue);
            }
        `}}function Y1(n){const{inputs:e,backend:t,attrs:s}=n,{image:o,transforms:r}=e,{interpolation:a,fillMode:c,fillValue:i,outputShape:l}=s,[u,d,p,h]=o.shape,[f,g]=l??[d,p],x=[u,f,g,h],m=new q1(d,p,a,c,i,x);return t.runWebGLProgram(m,[o,r],"float32")}const Q1={kernelName:xu,backendName:"webgl",kernelFunc:Y1};function Z1(n){const{inputs:e,attrs:t,backend:s}=n,{axis:o}=t,{x:r}=e;Ye(r,"unique"),console.warn("WARNING: ","UI might be locked temporarily as data is being downloaded");const a=s.readSync(r.dataId),{outputValues:c,outputShape:i,indices:l}=jh(a,o,r.shape,r.dtype);return[s.makeTensorInfo(i,r.dtype,c),s.makeTensorInfo([l.length],"int32",l)]}const J1={kernelName:gu,backendName:"webgl",kernelFunc:Z1};function ev(n){const{inputs:e,backend:t,attrs:s}=n,{value:o}=e;let{axis:r}=s;r<0&&(r+=o.shape.length);const a=o,c=a.shape.length,i=o.shape[r],l=new Array(c-1);let u=0;for(let g=0;g<c;g++)g!==r&&(l[u++]=a.shape[g]);const d=[],p=new Array(c).fill(0),h=a.shape.slice();h[r]=1;const f=new Array(i);for(let g=0;g<f.length;g++){p[r]=g;const x=st({inputs:{x:a},backend:t,attrs:{begin:p,size:h}}),m=R({inputs:{x},backend:t,attrs:{shape:l}});f[g]=m,d.push(x)}return d.forEach(g=>t.disposeIntermediateTensorInfo(g)),f}const tv={kernelName:Cu,backendName:"webgl",kernelFunc:ev};class nv{constructor(e,t){this.variableNames=["x","segmentIds"];const s=e.windowSize,o=e.batchSize,r=e.inSize,a=e.numSegments,c=a*Math.ceil(r/s);this.outputShape=[o,c];const i="0.0",l="sumValue",u=Math.floor(s/4)*4,d=s%4,p=`
        sumValue += dot(values, segFilter);
    `;let h="";r%s>0&&(h=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return initializationValue;
        }
      `);let f="";r%s>0&&(f=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return -1.0;
        }
      `),this.userCode=`
      const float initializationValue = ${i};

      float getValue(int batch, int inIdx) {
        ${h}
        return getX(batch, inIdx);
      }

      float getSegmentIdAtIndex(int inIdx) {
        ${f}
        return getSegmentIds(inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = int(floor(float(outIdx) / float(
          ${a})) * float(${s}));
        int currentSeg = int(mod(float(outIdx), float(${a})));

        float sumValue = 0.0;

        for (int i = 0; i < ${u}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 3)) == currentSeg ? 1 : 0
          );

          ${p}
        }

        int inIdx = inOffset + ${u};
        if (${d===1}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          int inIdxSeg = int(getSegmentIdAtIndex(inIdx));

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            0,
            0,
            0
          );

          ${p}
        } else if (${d===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
              0,
              0
          );

          ${p}
        } else if (${d===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            0
          );

          ${p}
        }
        setOutput(${l});
      }
    `}}function sv(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,segmentIds:r}=e,{numSegments:a}=s,c=o.shape.length,i=[];let l=0;const u=se([l],c);let d=o;u!=null&&(d=H({inputs:{x:o},backend:t,attrs:{perm:u}}),i.push(d),l=oe(1,c)[0]);const p=nr(d.shape,l,a),h=E([d.shape[l]]),f=R({inputs:{x:d},backend:t,attrs:{shape:[-1,h]}});i.push(f);const g=zn(o.dtype),x=(b,v,T,S,I)=>{const A=b.shape[0],k=b.shape[1],F=tr(k,I),P={windowSize:F,inSize:k,batchSize:A,numSegments:I},pe=new nv(P,v),Q=t.compileAndRun(pe,[b,T],S);if(i.push(Q),Q.shape[1]===I)return Q;const ee=ai({backend:t,attrs:{start:0,stop:I,step:1,dtype:"float32"}}),re=ii({inputs:{x:ee},backend:t,attrs:{reps:[k/F]}});return i.push(ee),i.push(re),x(Q,v,re,S,I)},m=x(f,"unsortedSegmentSum",r,g,a),C=R({inputs:{x:m},backend:t,attrs:{shape:p}});let $=C;if(u!=null){i.push(C);const b=$n(u);$=H({inputs:{x:$},backend:t,attrs:{perm:b}})}return i.forEach(b=>t.disposeIntermediateTensorInfo(b)),$}const ov={kernelName:$u,backendName:"webgl",kernelFunc:sv};const rv=[Uf,zf,Kf,Yf,Zf,tm,sm,rm,lm,dm,fm,gm,bm,Rm,Tm,Nm,Am,Pm,Lm,Bm,Gm,Ym,Zm,nx,ox,ux,px,xx,yf,$x,Rx,Ex,Fx,Lx,Bx,Wx,Gx,Kx,Yx,Jx,tg,sg,rg,cg,ug,fg,xg,$g,wg,Rg,Eg,Og,_g,Bg,Ug,Gg,Hg,Kg,qg,Qg,Jg,sC,aC,lC,dC,fC,gC,vC,yC,Rf,TC,wx,kC,DC,_C,Tf,MC,zC,XC,YC,JC,s0,a0,u0,f0,g0,$0,I0,y0,T0,A0,D0,P0,L0,B0,G0,K0,Q0,r$,kf,l$,p$,m$,C$,ax,v$,I$,y$,E$,O$,Nf,F$,_$,V$,M$,W$,ix,t$,z$,j$,Z$,Of,nb,rb,lb,pb,xb,Cb,vb,Rb,Tb,kb,Db,_b,Mb,Gb,Kb,Yb,jm,s$,Jb,t1,s1,r1,i1,l1,d1,h1,m1,C1,b1,w1,R1,T1,N1,A1,D1,n$,Bf,_1,B1,W1,z1,j1,Q1,Mf,J1,tv,ov,w$];for(const n of rv)bu(n);const Mv=Object.freeze(Object.defineProperty({__proto__:null,GPGPUContext:Lt,MathBackendWebGL:Rt,forceHalfFloat:Oa,gpgpu_util:nh,setWebGLContext:Pr,version_webgl:wf,webgl:If,webgl_util:cp},Symbol.toStringTag,{value:"Module"}));export{Sr as $,un as A,Vo as B,Mo as C,Wo as D,Bo as E,Uo as F,ud as G,Lo as H,_o as I,Po as J,Fo as K,Do as L,Oo as M,Au as N,Eu as O,Nu as P,ku as Q,Du as R,ko as S,Gd as T,Fu as U,Ou as V,Hn as W,Zu as X,sr as Y,Ju as Z,od as _,Ao as a,Rt as a$,ad as a0,ju as a1,bd as a2,vd as a3,wd as a4,Id as a5,To as a6,yd as a7,Nd as a8,kd as a9,Iv as aA,Rv as aB,yv as aC,Sv as aD,Tv as aE,Ev as aF,Nv as aG,kv as aH,Ov as aI,dv as aJ,Dv as aK,Fv as aL,Pv as aM,_v as aN,Lv as aO,Vv as aP,Bv as aQ,Av as aR,wf as aS,Vu as aT,yu as aU,iv as aV,Kd as aW,If as aX,nh as aY,cp as aZ,Oa as a_,Ad as aa,Go as ab,vo as ac,po as ad,_d as ae,Vd as af,Md as ag,Wd as ah,zd as ai,Hd as aj,Xd as ak,Ku as al,cv as am,hv as an,fv as ao,pv as ap,mv as aq,lv as ar,xv as as,gv as at,Cv as au,$v as av,bv as aw,vv as ax,wv as ay,uv as az,_e as b,Pr as b0,Lt as b1,Mv as b2,Y as c,Wu as d,J as e,Zn as f,Qn as g,Jn as h,dn as i,Eo as j,No as k,Ed as l,pt as m,Tu as n,Uu as o,qn as p,wo as q,Ae as r,Td as s,Tr as t,ur as u,ar as v,qt as w,Xu as x,Gu as y,ht as z};
