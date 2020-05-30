var oProgressLayer=null;
function SetBusy(){
for(var iCnt=0;iCnt<document.all.length;iCnt++){
try{document.all[iCnt].oldCursor=document.all[iCnt].style.cursor;
document.all[iCnt].style.cursor='wait';}catch(e){;}
try{document.all[iCnt].oldonmousedown=document.all[iCnt].onmousedown;
document.all[iCnt].onmousedown=function(){return false;}}catch(e){;}
try{document.all[iCnt].oldonclick=document.all[iCnt].onclick;
document.all[iCnt].onclick=function(){return false;}}catch(e){;}
try{document.all[iCnt].oldonmouseover=document.all[iCnt].onmouseover;
document.all[iCnt].onmouseover=function(){return false;}}catch(e){;}
try{document.all[iCnt].oldonmousemove=document.all[iCnt].onmousemove;
document.all[iCnt].onmousemove=function(){return false;}}catch(e){;}
try{document.all[iCnt].oldonkeydown=document.all[iCnt].onkeydown;
document.all[iCnt].onkeydown=function(){return false;}}catch(e){;}
try{document.all[iCnt].oldoncontextmenu=document.all[iCnt].oncontextmenu;
document.all[iCnt].oncontextmenu=function(){return false;}}catch(e){;}
try{document.all[iCnt].oldonselectstart=document.all[iCnt].onselectstart;
document.all[iCnt].onselectstart=function(){return false;}}catch(e){;}
}
}

function ReleaseBusy(){
for(var iCnt=0;iCnt<document.all.length;iCnt++){
try{document.all[iCnt].style.cursor=document.all[iCnt].oldCursor;}catch(e){;}
try{document.all[iCnt].onmousedown=document.all[iCnt].oldonmousedown;}catch(e){;}
try{document.all[iCnt].onclick=document.all[iCnt].oldonclick;}catch(e){;}
try{document.all[iCnt].onmouseover=document.all[iCnt].oldonmouseover;}catch(e){;}
try{document.all[iCnt].onmousemove=document.all[iCnt].oldonmousemove;}catch(e){;}
try{document.all[iCnt].onkeydown=document.all[iCnt].oldonkeydown;}catch(e){;}
try{document.all[iCnt].oncontextmenu=document.all[iCnt].oldoncontextmenu;}catch(e){;}
try{document.all[iCnt].onselectstart=document.all[iCnt].oldonselectstart;}catch(e){;}
}
}

function HideProgressInfo(){
if(oProgressLayer){
ReleaseBusy();
oProgressLayer.removeNode(true);
oProgressLayer=null;
}
}

function ShowProgressInfo(){
if(oProgressLayer) return;
oProgressLayer=document.createElement('DIV');
with(oProgressLayer.style){
width='230px';
height='70px';
position='absolute';
left=(document.body.clientWidth-230)>>1;
top=(document.body.clientHeight-70)>>1;
backgroundColor='buttonFace';
borderLeft='solid 1px silver';
borderTop='solid 1px silver';
borderRight='solid 1px gray';
borderBottom='solid 1px gray';
fontWeight='700';
fontSize='13px';
zIndex='999';
}
oProgressLayer.innerHTML='<table border="0" cellspacing="0" cellpadding="0" width="100%" height="100%">'+
'<tr>'+
'<td align="center" valign="middle">'+
'<img src="../static/img/wait.jpg" border="0" align="absmiddle" /> </td> </table>';
document.body.appendChild(oProgressLayer);
SetBusy();
}
