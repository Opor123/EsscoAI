const sideNav=document.querySelector('.sideNav');
const navLink=document.querySelectorAll('.nav-link')
const input=document.getElementById('chat-input');
const chat=document.getElementById('chatbox')

const API_BASE = window.__ESSCO_API_BASE__ || '/api';

let isLoading=false;
let conversations=[]

const uE1=document.getElementById('adminUser');
const pE1=document.getElementById('adminPass');
const saveCredsBtn=document.getElementById('saveAdminCreds');
const retrainBtn=document.getElementById('btnRetrain');
const retrainMsg=document.getElementById('retrainMsg');

const featureLink=document.querySelector('.nav-link[href="#feature"]')
const featureSection=document.getElementById('feature')

const mainEl=document.querySelector('main');

function showFunction(){
    if(!featureSection)return;
    featureSection.style.display='block';
    if(mainEl) mainEl.style.visibility='hidden';
    refreshStatus()
}
function hideFeature(){
    if(!featureSection)return;
    featureSection.style.display='none';
    if(mainEl)mainEl.style.visibility='visible';
}
//Generate Unique ID
function generateSessionId(){
    return `session_${Date.now()}_${Math.random().toString(36).slice(2,9)}`;
}

let currentSessionId=localStorage.getItem("ESSCO_SESSION_ID")||generateSessionId();
localStorage.setItem("ESSCO_SESSION_ID",currentSessionId);

//Initialize app
async function init(){
    if(localStorage.getItem("ESSCO_SIDENAV_COLLAPSED")==='1'){
        document.body.classList.add('sideNav-collapse');
    }
    await loadConversations();
    setupEventListeners();

    try{
        await refreshStatus();
    }catch (e){
        console.warn('Status refresh failed on init:',e)
    }

    input.focus();
}

document.getElementById('chat-form').addEventListener('submit',(e)=>{
    e.preventDefault();
    if(!isLoading)sendMsg();
})

function setupEventListeners(){
    document.querySelector('.close-open').addEventListener('click',(e)=>{
        e.preventDefault();
        document.body.classList.toggle('sideNav-collapse');
        localStorage.setItem('ESSCO_SIDENAV_COLLAPSED',
            document.body.classList.contains('sideNav-collapse')?'1':'0'
            )
    });

    const newChatLink=document.querySelector('.el .nav-link[href="#"]');

    if(newChatLink){
        newChatLink.addEventListener('click',(e)=>{
            e.preventDefault();
            hideFeature();
            startNewChat();
        })
    }


}

async function loadConversations(){
    hideFeature();
    try{
        const res=await fetch(`${API_BASE}/conversations?limit=50`,{
            headers:{'X-Session-Id':currentSessionId},
            cache:"no-store"
        });
        if(!res.ok) throw new Error('Failed to load conversations');

        conversations=await res.json();
        renderConversations();
    }catch(e){
        console.error('Error loading conversations: ',e);
    }
}

function renderConversations(){
    const container=document.querySelector('.sideNav');

    const oldList=document.getElementById('conversation-list');
    if (oldList) oldList.remove();

    const listDiv=document.createElement('div');
    listDiv.id='conversation-list';
    listDiv.className='conversation-list';

    if(conversations.length===0){
        listDiv.innerHTML='<div class="no-conversations">No conversations yet</div>'
    }else{
        conversations.forEach(con=>{
            const item=document.createElement('div');
            item.className='conversation-item';
            if (con.session_id===currentSessionId) {
                item.classList.add('active');
            }

            item.innerHTML=`
            <div class="conversation-content" data-session="${con.session_id}">
                <div class="conversation-title">${escapeHtml(con.title)}</div>
                <div class="conversation-preview">${escapeHtml(con.preview)}</div>
                <div class="conversation-meta">${formatTime(con.last_message_time)} ${con.message_count} msgs</div>
            </div>
            <button class="delete-conversation" data-session="${con.session_id}" title="Delete">
                <i class="fi fi-rr-trash"></i>
</button>
            `;

            item.querySelector('.conversation-content').addEventListener('click',()=>{
                loadConversation(con.session_id);
            })

            item.querySelector('.delete-conversation').addEventListener('click',(e)=>{
                e.stopPropagation();
                deleteConversation(con.session_id);
            });

            listDiv.appendChild(item);
        });
    }

    const bottomLine=container.querySelector('.bottom-line');
    bottomLine.after(listDiv);

    ensureConversationSearch();
}

function ensureConversationSearch(){
    if (document.querySelector('.conversation-search')) return;
    addConversationSearch();
}

async function loadConversation(sessionId){
    hideFeature();

    try{
        const res=await fetch(`${API_BASE}/conversations/${sessionId}?limit=200&offset=0`,{
            headers:{'X-Session-Id':currentSessionId}
        });
        if(!res.ok) throw new Error('Failed to load conversations');

        const data=await res.json();

        chat.innerHTML='';

        const welcome=document.getElementById('welcome');
        if(welcome) welcome.remove();

        currentSessionId=sessionId;
        localStorage.setItem("ESSCO_SESSION_ID",currentSessionId);

        data.messages.forEach(msg=>{
            if(msg.user_message){
                addMsg(msg.user_message,'user',false,msg.timestamp);
            }
            if(msg.bot_answer){
                addMsg(msg.bot_answer,'bot',false,msg.timestamp);
            }
        });

        document.querySelectorAll('.conversation-item').forEach(item=>{
            item.classList.remove('active');
            if(item.querySelector(`[data-session="${sessionId}"]`)){
                item.classList.add('active');
            }
        });

        input.focus();

    }catch(e){
        console.error('Error loading conversations:',e);
        showToast('Failed to load conversation','error');
    }
}

async function deleteConversation(sessionId){
    if(!confirm('Delete this conversation?')) return;
    try{
        const res=await fetch(`${API_BASE}/conversations/${sessionId}`,{
            method: 'DELETE',
            headers:{'X-Session-Id':currentSessionId}
        });

        if(!res.ok) throw new Error('Failed to delete conversation');

        if(sessionId===currentSessionId){
            startNewChat();
        }
        await loadConversations();
    }catch(error){
        console.error('Error deleting conversation:',error);
        showToast('Failed to delete conversation','error');
    }
}

function startNewChat(){
    currentSessionId=generateSessionId();
    localStorage.setItem("ESSCO_SESSION_ID",currentSessionId);
    chat.innerHTML='';

    if(!document.getElementById('welcome')){
        const welcome=document.createElement('h1');
        welcome.id='welcome';
        welcome.textContent='Welcome to Essco';
        document.querySelector('main').insertBefore(welcome,chat);
    }

    document.querySelectorAll('.conversation-item').forEach(item=>{
        item.classList.remove('active');
    });

    input.value='';
    input.focus();
    loadConversations().catch(()=>{});

}
const submitBtn=document.getElementById('submitMsg');
async function sendMsg(){
    const text=input.value.trim();
    if(!text||isLoading)return;

    isLoading=true;
    submitBtn&&(submitBtn.disabled=true);

    addMsg(text,'user');
    input.value='';

    const loadingId=addTypingIndicator();

    try{
        // Use the new LLM-enhanced endpoint
        const res=await fetch(`${API_BASE}/chat-llm`,{
            method: 'POST',
            headers:{
                'Content-Type':'application/json',
                'X-Session-Id':currentSessionId
            },
            body:JSON.stringify({
                message:text,
                session_id:currentSessionId,
            }),
        });

        if(res.status==429){
            removeMsg(loadingId);
            addMsg("Rate limit hit. Please try again in a minute.",'bot');
            return;
        }

        if(!res.ok) {
            if(res.status==503){
                removeMsg(loadingId);
                addMsg("Service is warming up or temporarily unavailable. Please try again shortly.", "bot");
                return;
            }
            throw new Error(`HTTP ${res.status}: ${res.statusText}`)
        };

        const data=await res.json();
        const ans=data.answer||data.message||"No response Received";

        const welcome=document.getElementById('welcome');
        if(welcome){
            welcome.style.animation="fadeOut 0.3s ease";
            setTimeout(()=>welcome.remove(),300);
        }

        removeMsg(loadingId);
        const mid=addMsg(ans,'bot');

        // Show LLM badge if it was used
        if(data.meta?.llm_active){
            const llmBadge=document.createElement("div");
            llmBadge.className='llm-badge';
            llmBadge.innerHTML='<i class="fi fi-rr-sparkles"></i> AI Enhanced';
            llmBadge.title=data.meta?.reason || 'Enhanced with AI for better response';
            document.getElementById(mid)?.appendChild(llmBadge);
        }

        // Show confidence badge
        if(typeof(data.conf)==='number'){
            const badge=document.createElement("div");
            const lvl=data.conf>=0.7?'high':(data.conf>=0.4?'medium':'low');
            badge.className=`confidence-badge confidence-${lvl}`;
            badge.textContent=`${Math.round(data.conf*100)}% ${data?.meta?.confidence_label||'confidence'}`;
            document.getElementById(mid)?.appendChild(badge);
        }

        // Show mode indicator (optional, for debugging)
        if(data.meta?.mode){
            const modeInfo=document.createElement("div");
            modeInfo.className='mode-info';
            const modeText = {
                'llm_enhanced': 'AI-Enhanced',
                'retrieval_only': 'Knowledge Base',
                'retrieval_fallback': 'KB Fallback'
            }[data.meta.mode] || data.meta.mode;
            modeInfo.textContent=modeText;
            document.getElementById(mid)?.appendChild(modeInfo);
        }

        await loadConversations();
    }catch(error){
        console.error('API Error:',error);
        removeMsg(loadingId);
        addMsg(`Error: Couldn't connect to API - ${error.message}`,'bot');
    }finally {
        isLoading=false;
        submitBtn&&(submitBtn.disabled=false);
        input.focus();
    }
}

async function checkLLMStatus(){
    try{
        const res=await fetch(`${API_BASE}/llm-status`,{
            headers:{'X-Session-Id':currentSessionId},
            cache:"no-store"
        });
        if(!res.ok) return null;

        const status=await res.json();
        console.log('LLM Status:', status);

        // Show status in console
        if(status.available){
            console.log(`✓ AI Enhancement enabled (${status.model})`);
            console.log(`  Threshold: ${status.confidence_threshold}, Max tokens: ${status.max_tokens}`);
        }else{
            console.log('ℹ AI Enhancement disabled:', status.reason);
        }

        return status;
    }catch(e){
        console.warn('Could not check LLM status:',e);
        return null;
    }
}

async function refreshStatus(){
    const el=document.getElementById('statusText');
    try{
        const res=await fetch(`${API_BASE}/status`,{
            headers:{'X-Session-Id':currentSessionId},
            cache:'no-store'
        });
        let data={}
        try{data=await res.json();}catch{/*ignore parse error*/}
        if (el) {
            el.textContent = `Loaded: ${data.qa_system_loaded ?? 'n/a'} | QA pairs: ${data.total_qa_pairs ?? 'n/a'}`;
        }

        const adminLink = document.getElementById('adminLink');
        if (adminLink) {
            adminLink.style.display = data.admin_link_enabled ? 'flex' : 'none';
        }
    }catch{
        if (el) el.textContent='Failed to load status';
    }
}


document.getElementById('refreshStatus')?.addEventListener('click',refreshStatus);


function addMsg(text,sender,animate=true,tsSec=null){
    const div=document.createElement('div');
    div.classList.add('msg',sender);
    const content=document.createElement('div');
    content.className='msg-content';
    content.textContent=text;

    const stamp=document.createElement('div');
    stamp.className='msg-timestamp';
    const when = tsSec?new Date(tsSec*1000):new Date();

    stamp.textContent=when.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});

    div.appendChild(content);
    div.appendChild(stamp);

    const messageId=`msg-${Date.now()}-${Math.random().toString(36).slice(2,9)}`;
    div.id=messageId;

    chat.appendChild(div);
    chat.scrollTop=chat.scrollHeight;

    return messageId;
}

function removeMsg(messageId){
    const ele=document.getElementById(messageId);
    if(ele){
        ele.style.animation='fadeOut 0.2s ease';
        setTimeout(()=>ele.remove(),200);
    }
}

function escapeHtml(text){
    const div=document.createElement('div');
    div.textContent=text;
    return div.innerHTML;
}

function formatTime(timestamp){
    const date=new Date(timestamp*1000);
    const now=new Date();
    const diff=now-date;

    if(diff<60000) return 'Just Now';

    if(diff<3600000) {
        const mins = Math.floor(diff / 60000);
        return `${mins}m ago`;
    }

    if(diff<86400000){
        const hours = Math.floor(diff / 3600000);
        return `${hours}h ago`;
    }

    if(diff<604800000){
        const days=Math.floor(diff/86400000);
        return `${days}d ago`;

    }

    return date.toLocaleDateString();
}

const style=document.createElement('style');
style.textContent=`
@keyframes fadeOut{
    from{opacity: 1;transform:scale(1);}
    to{opacity: 0;transform:scale(0.95);}
}
`;
document.head.appendChild(style);

if(document.readyState=='loading'){
    document.addEventListener('DOMContentLoaded',()=>{
        init();
        checkLLMStatus();
    });
}else{
    init();
    checkLLMStatus();
}

// Show Feature panel + load status
if (featureLink&&featureSection){
    featureLink.addEventListener('click', (e)=>{
       e.preventDefault();
       const visible = getComputedStyle(featureSection).display!=='none';
       if(visible){hideFeature();}else{showFunction()}
    });
}



//FeedBack
let fbLabel='neutral';
document.getElementById('fbUp')?.addEventListener('click',()=>fbLabel='up');
document.getElementById('fbDown')?.addEventListener('click',()=>fbLabel='down');
document.getElementById('fbSend')?.addEventListener('click',async ()=>{
    const note=document.getElementById('fbNote').value.trim();
    const payload={label:fbLabel,user_comment:note||undefined,session_id:currentSessionId};
    const msg=document.getElementById('fbMsg');
    try{
        const res=await fetch(`${API_BASE}/feedback`,{
            method: 'POST',
            headers:{'Content-Type':'application/json','X-Session-Id':currentSessionId},
            body:JSON.stringify(payload)
        });
        if (res.status===429){
            msg.textContent='Too many requests. Try later.';
            return;
        }
        if(!res.ok) throw new Error(await res.text());
        msg.textContent='Thanks! Feedback recorded.';
        document.getElementById('fbNote').value='';
        fbLabel='neutral';
    }catch (e) {
        msg.textContent='Failed to get feedback';
    }
});

function addConversationSearch(){
    const searchInput=document.createElement('input');
    searchInput.type='text';
    searchInput.placeholder='Search conversation...';
    searchInput.className='conversation-search';

    searchInput.addEventListener('input',(e)=>{
        const query=e.target.value.toLowerCase();
        document.querySelectorAll('.conversation-item').forEach(item=>{
            const title=item.querySelector('.conversation-title').textContent.toLowerCase();
            const preview=item.querySelector('.conversation-preview').textContent.toLowerCase();
            item.style.display=(title.includes(query)||preview.includes(query))?'flex':'none';
        });
    });

    const firstEl=document.querySelector('.sideNav .el');
    firstEl.parentNode.insertBefore(searchInput,firstEl);
}

// Prefer sessionStorage for credentials
(
    function initAdminBasic(){
        const u=sessionStorage.getItem('ESSCO_ADMIN_USER') || '';
        const p=sessionStorage.getItem('ESSCO_ADMIN_PASS') || '';
        if (u) uE1.value=u;
        if (p) pE1.value=p;
        if (u&&p) retrainBtn.disabled=false;
    }
    )();

saveCredsBtn?.addEventListener('click',()=>{
    const u=(uE1.value||"").trim();
    const p=(pE1.value||'').trim();
    if(!u||!p){ showToast('Enter username and password','info'); return;}
    sessionStorage.setItem('ESSCO_ADMIN_USER',u);
    sessionStorage.setItem('ESSCO_ADMIN_PASS',p);
    retrainBtn.disabled=false;
});

retrainBtn?.addEventListener('click',async ()=>{
    const u = sessionStorage.getItem('ESSCO_ADMIN_USER') || '';
    const p = sessionStorage.getItem('ESSCO_ADMIN_PASS') || '';

    if(!u||!p) { retrainMsg.textContent = 'Set admin account first.'; return; }

    const basic=btoa(`${u}:${p}`);
    retrainMsg.textContent='Retraining...';
    retrainBtn.disabled=true;
    try{
        const res=await fetch(`${API_BASE}/retrain`,{
            method: 'POST',
            headers:{'Authorization':`Basic ${basic}`,'X-Session-Id':currentSessionId},
        });
        const data=await res.json().catch(()=>({}));
        if (!res.ok) throw new Error(data?.detail || res.statusText);
        retrainMsg.textContent=data?.message||'Retrained successfully';
    }catch(e){
        retrainMsg.textContent=`Failed: ${e.message}`;
    }finally {
        retrainBtn.disabled=false;
    }
});

function addTypingIndicator(){
    const wrap=document.createElement('div');
    wrap.className='msg bot typing-indicator';
    wrap.innerHTML=`
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
    `
    const id =`msg-${Date.now()}-${Math.random().toString(36).slice(2,9)}`;
    wrap.id=id;
    chat.appendChild(wrap);
    chat.scrollTop=chat.scrollHeight;
    return id;

}

function showToast(message,type='info'){
    const t=document.createElement('div');
    t.className=`toast toast-${type}`;
    t.textContent=message;
    document.body.appendChild(t);
    requestAnimationFrame(() => t.classList.add('show'));
    setTimeout(()=>{
        t.classList.remove('show');
        setTimeout(()=>t.remove(),300);
    },3000);
}

const adminLink = document.getElementById('adminLink');
const adminSection = document.getElementById('admin-dashboard');

async function loadAdminDashboard(){
    try{
        const res = await fetch(`${API_BASE}/admin/dashboard`, {
            // no headers: browser will show basic auth popup on 401
            credentials: 'include',
        });
        if(!res.ok){
            console.error('Admin dashboard error', res.status, res.statusText);
            showToast('Failed to load admin dashboard','error');
            return;
        }
        const d = await res.json();

        document.getElementById('adm-total-conv').textContent = d.total_conversations;
        document.getElementById('adm-total-msg').textContent = d.total_messages;
        document.getElementById('adm-last24-msg').textContent = d.last_24h_messages;

        document.getElementById('adm-qa-loaded').textContent = d.qa_system_loaded ? 'Yes' : 'No';
        document.getElementById('adm-qa-pairs').textContent = d.total_qa_pairs ?? '-';

        document.getElementById('adm-llm-avail').textContent = d.llm_available ? 'Yes' : 'No';
        document.getElementById('adm-llm-enabled').textContent = d.llm_enabled ? 'Yes' : 'No';
        document.getElementById('adm-llm-model').textContent = d.llm_model || '-';

        document.getElementById('adm-fb-total').textContent = d.feedback_total;
        document.getElementById('adm-fb-up').textContent = d.feedback_up;
        document.getElementById('adm-fb-down').textContent = d.feedback_down;
        document.getElementById('adm-fb-neutral').textContent = d.feedback_neutral;
    }catch(e){
        console.error('Admin dashboard load error', e);
        showToast('Failed to load admin dashboard','error');
    }
}

if (adminLink && adminSection){
    adminLink.addEventListener('click', (e)=>{
        e.preventDefault();
        const visible = getComputedStyle(adminSection).display !== 'none';
        adminSection.style.display = visible ? 'none' : 'block';
        if (!visible){
            adminLink.classList.add('active');
            loadAdminDashboard();
        } else {
            adminLink.classList.remove('active');
        }
    });
}
