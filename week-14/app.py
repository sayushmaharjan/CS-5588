"""Cinematic Memory — Streamlit UI v4 (no sidebar, single ambient/music, previews)"""
import streamlit as st, os, sys, time, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

st.set_page_config(page_title="Cinematic Memory", page_icon="🎬",
    layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#f4f5fc;}
[data-testid="stSidebar"]{display:none;}
.stButton>button{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;font-weight:600;
  border:none;border-radius:10px;padding:10px 24px;width:100%;font-size:.95rem;
  box-shadow:0 4px 14px rgba(99,102,241,.3);transition:all .2s;}
.stButton>button:hover{opacity:.88;transform:translateY(-1px);}
.stButton>button:disabled{background:#d1d5db;box-shadow:none;transform:none;}
[data-testid="stFileUploader"]{border:2px dashed #c7d2fe!important;border-radius:12px;}
.card{background:#fff;border:2px solid #eef0f8;border-radius:14px;padding:18px;margin-bottom:12px;}
.card-sel{border-color:#6366f1;background:#eef2ff;}
</style>""", unsafe_allow_html=True)

MEDIA_T=["jpg","jpeg","png","webp","mp4","mov","avi","mkv","webm"]
AUDIO_T=["mp3","wav","m4a","ogg","flac","aac"]
ACT_C={"setup":"#059669","peak":"#d97706","reflection":"#7c3aed"}
EMO={"joyful":"😄","nostalgic":"🥺","reflective":"🤔","sad":"😢",
     "excited":"🤩","neutral":"😐","celebratory":"🎉","tender":"🥰"}

def fmt(s): m=int(s)//60; return f"{m}m {int(s)%60:02d}s" if m else f"{int(s)}s"

def save_files(files, d):
    os.makedirs(d, exist_ok=True)
    paths = []
    for f in files:
        p = os.path.join(d, f.name)
        with open(p, "wb") as out:
            out.write(f.getbuffer())
        paths.append(p)
    return paths

def pills(states):
    cols=st.columns(4)
    for col,(n,lbl,s) in zip(cols,states):
        bg={"active":"#6366f1","done":"#10b981"}.get(s,"#e5e7eb")
        fg="white" if s in ("active","done") else "#9ca3af"
        pre="✓ " if s=="done" else ""
        col.markdown(f'<div style="text-align:center;padding:8px;background:{bg};border-radius:10px;'
            f'color:{fg};font-size:.8rem;font-weight:600">{pre}Step {n}: {lbl}</div>',unsafe_allow_html=True)
    st.markdown("<div style='margin:20px 0 0'></div>",unsafe_allow_html=True)

DEFS={"stage":"upload","pre_result":None,"pipeline_result":None,"run_id":None,
      "media_paths":[],"audio_paths":[],"voice_ref_path":None,"selected_version":0,
      "music_mood":"cinematic","ambient_scene":"nature",
      "nar_vol":1.0,"mus_vol":0.35,"amb_vol":0.20,
      "event_hint":"","user_script_input":"","target_duration_s":60.0,
      "script_mode":"cinematic"}
for k,v in DEFS.items():
    if k not in st.session_state: st.session_state[k]=v

# Hero
st.markdown('<div style="text-align:center;padding:24px 0 6px">'
    '<h1 style="font-size:2.6rem;font-weight:700;margin:0;background:linear-gradient(135deg,#6366f1,#8b5cf6);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent">🎬 Cinematic Memory</h1>'
    '<p style="color:#6b7280;font-size:1rem;margin:6px 0 0">Transform your memories into a cinematic documentary</p></div>',
    unsafe_allow_html=True)

# ══════════════════════ UPLOAD ═══════════════════════════════════════════════
if st.session_state.stage=="upload":
    pills([("1","Upload","active"),("2","Analyze",""),("3","Customize",""),("4","Export","")])
    c1,c2=st.columns(2,gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("#### 📸 Photos & Videos")
            mf=st.file_uploader("media",type=MEDIA_T,accept_multiple_files=True,label_visibility="collapsed")
            if mf:
                st.success(f"✓ {len(mf)} file(s) ready")
                imgs=[f for f in mf if f.type.startswith("image")][:4]
                if imgs:
                    pcols=st.columns(min(len(imgs),4))
                    for pc,img in zip(pcols,imgs): pc.image(img,use_container_width=True)
    with c2:
        with st.container(border=True):
            st.markdown("#### ✍️ Your Script _(optional)_")
            usc=st.text_area("sc",height=90,label_visibility="collapsed",
                placeholder="Leave blank to auto-generate, or write your story here…")
            if usc and usc.strip():
                smode=st.radio("Script mode",
                    ["✨ Cinematic (AI reshapes your words)","📝 Exact (use your words as-is)"],
                    index=0, horizontal=True, label_visibility="collapsed")
                st.session_state.script_mode="exact" if "Exact" in smode else "cinematic"
            st.markdown("#### 🎙️ Voice Memos _(optional)_")
            af=st.file_uploader("audio",type=AUDIO_T,accept_multiple_files=True,label_visibility="collapsed")
            if af: st.success(f"✓ {len(af)} memo(s)")
            
            st.markdown("#### 🗣️ Narrator Voice Clone _(optional)_")
            vf=st.file_uploader("voice_ref",type=["wav"],accept_multiple_files=False,label_visibility="collapsed", help="Upload a short .wav file to clone the narrator's voice")
            if vf: st.success("✓ Voice clone reference loaded")

    with st.container(border=True):
        st.markdown("#### ⚙️ Settings")
        sc1,sc2,sc3,sc4=st.columns(4)
        with sc1:
            ev=st.selectbox("Event type",["","wedding","vacation","birthday","graduation","adventure","memorial","custom"])
            if ev=="custom": ev=st.text_input("Describe:")
        with sc2:
            dp=st.select_slider("Length",["30s","60s","90s","120s","Custom"],"60s")
            td={"30s":30,"60s":60,"90s":90,"120s":120}.get(dp,60)
            if dp=="Custom": td=st.number_input("Seconds",15,300,60,5)
        with sc3:
            provider_txt={"groq":"🟢 Groq (free)","anthropic":"🔵 Anthropic","template":"⚪ Template"}.get(cfg.LLM_PROVIDER,cfg.LLM_PROVIDER)
            key_ok=bool(cfg.GROQ_API_KEY.strip() if cfg.LLM_PROVIDER=="groq" else cfg.ANTHROPIC_API_KEY.strip())
            st.markdown(f"**AI:** {provider_txt}")
            st.caption("✅ Key configured" if key_ok else "⚠️ Set key in config.py")
        with sc4:
            st.markdown(f"**3 script versions** generated automatically")
            st.caption("Pick your favourite in Step 3")

    can=bool(mf or af or (usc and usc.strip()))
    if not can: st.info("📁 Upload at least one photo, video, voice memo, or type a script.")
    _,bc,_=st.columns([1.5,2,1.5])
    with bc:
        if st.button("🎬 Analyze & Generate Scripts",disabled=not can,use_container_width=True):
            rid=f"run_{int(time.time())}"
            udir=os.path.join(tempfile.gettempdir(),"cm",rid,"uploads")
            mp=save_files(mf or [],os.path.join(udir,"media"))
            ap=save_files(af or [],os.path.join(udir,"audio"))
            vp=save_files([vf] if vf else [],os.path.join(udir,"voice"))
            st.session_state.update({"run_id":rid,"media_paths":mp,"audio_paths":ap,"voice_ref_path":vp[0] if vp else None,
                "event_hint":ev,"user_script_input":usc,"target_duration_s":float(td),
                "stage":"processing","pre_result":None,"pipeline_result":None,
                "selected_version":0,"music_mood":"cinematic","ambient_scene":"nature"})
            st.rerun()

# ══════════════════════ PROCESSING ═══════════════════════════════════════════
elif st.session_state.stage=="processing":
    pills([("1","Upload","done"),("2","Analyze","active"),("3","Customize",""),("4","Export","")])
    st.markdown("## ⚙️ Analyzing & Generating Scripts")
    ph=st.empty(); lh=st.empty()
    if st.session_state.pre_result is None:
        msgs=[]
        def upd(msg,pct):
            msgs.append((msg,pct))
            with ph.container(): st.progress(pct/100); st.markdown(f"**{msg}**")
            with lh.container():
                for m,p in msgs[-5:]: st.caption(f"{'✅' if p>=85 else '⏳'} `{p:3d}%` — {m}")
        odir=os.path.join(tempfile.gettempdir(),"cm",st.session_state.run_id,"output")
        os.makedirs(odir,exist_ok=True)
        try:
            from pipeline.orchestrator import run_pipeline_pre
            pre=run_pipeline_pre(
                photo_video_paths=st.session_state.media_paths,
                audio_paths=st.session_state.audio_paths,
                output_dir=odir,
                event_hint=st.session_state.event_hint,
                user_script=st.session_state.user_script_input or None,
                target_duration_s=st.session_state.target_duration_s,
                script_mode=st.session_state.script_mode,
                progress_cb=upd,
            )
            st.session_state.pre_result=pre
            st.session_state.stage="editing"; st.rerun()
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback; st.code(traceback.format_exc())
            if st.button("← Start Over"): st.session_state.stage="upload"; st.rerun()

# ══════════════════════ EDITING ══════════════════════════════════════════════
elif st.session_state.stage=="editing":
    pills([("1","Upload","done"),("2","Analyze","done"),("3","Customize","active"),("4","Export","")])
    pre=st.session_state.pre_result
    svs=pre["script_versions"]
    sel=st.session_state.selected_version
    script=svs[sel]

    from pipeline.music_generation import MUSIC_MOODS
    from pipeline.ambient_sound import AMBIENT_SCENES
    from pipeline.narrative_engine import TONE_VARIANTS

    st.markdown(f"## 🎛️ Customize Your Documentary")
    st.caption(f"Target: {fmt(st.session_state.target_duration_s)}  ·  Estimated: {fmt(script.total_duration_s)}  ·  {len(script.beats)} beats")

    tab_sc,tab_mu,tab_am,tab_mix=st.tabs(["📖 Script Version","🎵 Music","🌊 Ambient","🎚️ Mix Levels"])

    # ── SCRIPT TAB ────────────────────────────────────────────────────────
    with tab_sc:
        st.markdown("### 📖 Choose Your Script Style")
        st.caption("3 versions were generated with different emotional tones. Preview and select.")
        for i,(sc,var) in enumerate(zip(svs,TONE_VARIANTS)):
            is_sel=(i==sel)
            with st.container(border=True):
                hc,bc=st.columns([4,1])
                with hc:
                    badge='<span style="background:#6366f1;color:#fff;padding:2px 8px;border-radius:20px;font-size:.72rem;font-weight:600;margin-left:8px">Selected</span>' if is_sel else ""
                    st.markdown(f'**{var["icon"]} {var["label"]}**{badge}', unsafe_allow_html=True)
                    st.caption(var["description"])
                with bc:
                    if not is_sel:
                        if st.button("Select",key=f"sv{i}",use_container_width=True):
                            st.session_state.selected_version=i; st.rerun()
                    else:
                        st.markdown('<p style="color:#6366f1;font-weight:600;text-align:center;padding-top:6px">✓ Active</p>',unsafe_allow_html=True)

                # Preview narration (opening beat)
                preview_txt=sc.beats[0].narration_text if sc.beats else ""
                st.markdown(f'<div style="border-left:3px solid #6366f1;padding:8px 14px;margin:6px 0;'
                    f'font-style:italic;color:#374151;font-size:.9rem;line-height:1.7">"{preview_txt[:250]}…"</div>',
                    unsafe_allow_html=True)
                
                if st.button("▶️ Play Audio", key=f"preview_vo_{i}"):
                    with st.spinner("Generating audio..."):
                        from pipeline.voice_synthesis import _generate_chatterbox_audio
                        pdir=os.path.join(tempfile.gettempdir(),"cm","previews")
                        os.makedirs(pdir, exist_ok=True)
                        tpath = os.path.join(pdir, f"vo_prev_{i}.wav")
                        _generate_chatterbox_audio(preview_txt, tpath, reference_audio_path=st.session_state.get("voice_ref_path"))
                        if os.path.exists(tpath): st.audio(tpath)

                with st.expander("▶ Read full script"):
                    full=""
                    for beat in sc.beats:
                        full+=f"\n**{EMO.get(beat.emotion.value,'🎭')} {beat.beat_id}** · {beat.act_phase.value.upper()}\n\n> {beat.narration_text}\n"
                    st.markdown(full)

    # ── MUSIC TAB ─────────────────────────────────────────────────────────
    with tab_mu:
        st.markdown("### 🎵 Choose Background Music")
        st.caption("One track plays across the entire video. Preview, then select.")
        cur_mood=st.session_state.music_mood
        cols=st.columns(4)
        for i,mk in enumerate(MUSIC_MOODS.keys()):
            info=MUSIC_MOODS[mk]; is_sel=(mk==cur_mood)
            with cols[i%4]:
                border="2px solid #6366f1" if is_sel else "2px solid #eef0f8"
                bg="#eef2ff" if is_sel else "#fff"
                st.markdown(f'<div style="background:{bg};border:{border};border-radius:12px;'
                    f'padding:14px;text-align:center;margin-bottom:8px">'
                    f'<div style="font-size:1.5rem">{info["label"].split()[0]}</div>'
                    f'<div style="font-weight:600;font-size:.85rem">{"".join(info["label"].split()[1:])}</div>'
                    f'<div style="color:#6b7280;font-size:.75rem;margin:4px 0">{info["description"]}</div>'
                    f'</div>',unsafe_allow_html=True)
                if st.button("▶️ Preview Music", key=f"prev_mus_{mk}", use_container_width=True):
                    with st.spinner("Generating..."):
                        from pipeline.music_generation import generate_single_music_track
                        pdir=os.path.join(tempfile.gettempdir(),"cm","previews")
                        pp=generate_single_music_track(mk, 15.0, pdir, f"prev_{mk}.wav")
                        if pp and os.path.exists(pp): st.audio(pp)
                if not is_sel:
                    if st.button("Select",key=f"ms{mk}",use_container_width=True):
                        st.session_state.music_mood=mk; st.rerun()
                else:
                    st.markdown('<p style="color:#6366f1;font-weight:600;text-align:center;font-size:.8rem">✓ Selected</p>',unsafe_allow_html=True)

        st.divider()
        st.markdown(f"**Selected: {MUSIC_MOODS[cur_mood]['label']}**  —  _{MUSIC_MOODS[cur_mood]['description']}_")

    # ── AMBIENT TAB ───────────────────────────────────────────────────────
    with tab_am:
        st.markdown("### 🌊 Choose Ambient Soundscape")
        st.caption("One ambient track plays softly underneath the entire video.")
        cur_sc=st.session_state.ambient_scene
        acols=st.columns(4)
        for i,sk in enumerate(AMBIENT_SCENES.keys()):
            info=AMBIENT_SCENES[sk]; is_sel=(sk==cur_sc)
            with acols[i%4]:
                border="2px solid #6366f1" if is_sel else "2px solid #eef0f8"
                bg="#eef2ff" if is_sel else "#fff"
                st.markdown(f'<div style="background:{bg};border:{border};border-radius:12px;'
                    f'padding:14px;text-align:center;margin-bottom:8px">'
                    f'<div style="font-size:1.5rem">{info["label"].split()[0]}</div>'
                    f'<div style="font-weight:600;font-size:.85rem">{"".join(info["label"].split()[1:])}</div>'
                    f'<div style="color:#6b7280;font-size:.75rem;margin:4px 0">{info["description"]}</div>'
                    f'</div>',unsafe_allow_html=True)
                if st.button("▶️ Preview Ambient", key=f"prev_amb_{sk}", use_container_width=True):
                    with st.spinner("Generating..."):
                        from pipeline.ambient_sound import generate_single_ambient_track
                        pdir=os.path.join(tempfile.gettempdir(),"cm","previews")
                        pp=generate_single_ambient_track(sk, 15.0, pdir, f"prev_{sk}.wav")
                        if pp and os.path.exists(pp): st.audio(pp)
                if not is_sel:
                    if st.button("Select",key=f"as{sk}",use_container_width=True):
                        st.session_state.ambient_scene=sk; st.rerun()
                else:
                    st.markdown('<p style="color:#6366f1;font-weight:600;text-align:center;font-size:.8rem">✓ Selected</p>',unsafe_allow_html=True)

        st.divider()
        st.markdown(f"**Selected: {AMBIENT_SCENES[cur_sc]['label']}**  —  _{AMBIENT_SCENES[cur_sc]['description']}_")

    # ── MIX TAB ───────────────────────────────────────────────────────────
    with tab_mix:
        st.markdown("### 🎚️ Audio Mix Levels")
        st.caption("Adjust relative volumes before final render.")
        mc1,mc2,mc3=st.columns(3)
        with mc1:
            nv=st.slider("🔊 Narration",0.0,1.0,st.session_state.nar_vol,0.05)
            st.session_state.nar_vol=nv
        with mc2:
            mv=st.slider("🎵 Music",0.0,1.0,st.session_state.mus_vol,0.05)
            st.session_state.mus_vol=mv
        with mc3:
            av=st.slider("🌊 Ambient",0.0,1.0,st.session_state.amb_vol,0.05)
            st.session_state.amb_vol=av

    st.divider()
    ba,_,bb=st.columns([1,2,1])
    with ba:
        if st.button("← Back",use_container_width=True): st.session_state.stage="upload"; st.rerun()
    with bb:
        if st.button("🎬 Generate Final Video",use_container_width=True):
            st.session_state.stage="finalizing"; st.rerun()

# ══════════════════════ FINALIZING ════════════════════════════════════════════
elif st.session_state.stage=="finalizing":
    pills([("1","Upload","done"),("2","Analyze","done"),("3","Customize","done"),("4","Export","active")])
    st.markdown("## 🎬 Assembling Your Documentary")
    ph=st.empty(); lh=st.empty()
    if st.session_state.pipeline_result is None:
        msgs=[]
        def upd2(msg,pct):
            msgs.append((msg,pct))
            with ph.container(): st.progress(pct/100); st.markdown(f"**{msg}**")
            with lh.container():
                for m,p in msgs[-4:]: st.caption(f"{'✅' if p>=100 else '⏳'} `{p:3d}%` — {m}")
        try:
            from pipeline.orchestrator import run_pipeline_finalize
            result=run_pipeline_finalize(
                pre_result=st.session_state.pre_result,
                selected_version=st.session_state.selected_version,
                music_mood=st.session_state.music_mood,
                ambient_scene=st.session_state.ambient_scene,
                narration_vol=st.session_state.nar_vol,
                music_vol=st.session_state.mus_vol,
                ambient_vol=st.session_state.amb_vol,
                voice_reference_path=st.session_state.voice_ref_path,
                progress_cb=upd2,
            )
            st.session_state.pipeline_result=result
            st.session_state.stage="results"; st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback; st.code(traceback.format_exc())
            if st.button("← Back to Customize"): st.session_state.stage="editing"; st.rerun()

# ══════════════════════ RESULTS ══════════════════════════════════════════════
elif st.session_state.stage=="results":
    from pipeline.music_generation import MUSIC_MOODS
    from pipeline.ambient_sound import AMBIENT_SCENES
    result=st.session_state.pipeline_result
    script=result["script"]
    pills([("1","Upload","done"),("2","Analyze","done"),("3","Customize","done"),("4","Export","done")])

    mi=MUSIC_MOODS.get(result.get("music_mood","cinematic"),MUSIC_MOODS["cinematic"])
    ai=AMBIENT_SCENES.get(result.get("ambient_scene","nature"),AMBIENT_SCENES["nature"])
    st.markdown(f'<div style="text-align:center;padding:12px 0 6px">'
        f'<h2 style="color:#6366f1;margin:0">"{script.title}"</h2>'
        f'<p style="color:#6b7280;margin:6px 0 0">{script.arc_summary}</p>'
        f'<p style="color:#9ca3af;font-size:.82rem;margin:4px 0">Music: {mi["label"]}  ·  Ambient: {ai["label"]}  ·  {fmt(script.total_duration_s)}</p></div>',
        unsafe_allow_html=True)

    timings=result.get("timings",{})
    tot=sum(v for k,v in timings.items() if isinstance(v,(int,float)) and k!="pre_total")
    m1,m2,m3,m4=st.columns(4)
    for col,val,lbl in zip([m1,m2,m3,m4],
            [len(script.beats),fmt(script.total_duration_s),len(result.get("visual_meta",{})),f"{tot:.0f}s"],
            ["Beats","Runtime","Media","Time"]):
        with col:
            with st.container(border=True):
                st.markdown(f'<div style="text-align:center;padding:8px"><div style="font-size:1.8rem;font-weight:700;color:#6366f1">{val}</div>'
                    f'<div style="font-size:.7rem;color:#9ca3af;text-transform:uppercase;letter-spacing:1px">{lbl}</div></div>',unsafe_allow_html=True)

    st.divider()
    t1,t2,t3=st.tabs(["🎬 Video & Audio","📖 Full Script","📊 Stats"])

    with t1:
        vp=result.get("output_video_path")
        if vp and os.path.exists(vp):
            st.video(vp)
            with open(vp,"rb") as f:
                st.download_button("⬇️ Download MP4",f,file_name=f"{script.title.replace(' ','_')}.mp4",mime="video/mp4")
        else:
            st.info("🎬 Video rendering requires your images. Audio tracks are available below.")

        st.markdown("---")
        cc1,cc2=st.columns(2)
        with cc1:
            st.markdown("**🔊 Narration Beats**")
            for beat in script.beats:
                nar=result["narration_audio"].get(beat.beat_id)
                with st.expander(f"{EMO.get(beat.emotion.value,'🎭')} {beat.beat_id} · {fmt(beat.duration_hint_s)}"):
                    st.caption(f"_{beat.narration_text[:160]}_")
                    if nar and os.path.exists(nar.audio_path): st.audio(nar.audio_path)
        with cc2:
            st.markdown(f"**🎵 Music — {mi['label']}**")
            mp=result.get("global_music_path")
            if mp and os.path.exists(mp): st.audio(mp)
            st.markdown(f"**🌊 Ambient — {ai['label']}**")
            ap=result.get("global_ambient_path")
            if ap and os.path.exists(ap): st.audio(ap)

    with t2:
        cur_act=None
        for beat in script.beats:
            if beat.act_phase.value!=cur_act:
                cur_act=beat.act_phase.value
                c=ACT_C.get(cur_act,"#6366f1")
                lbl={"setup":"ACT I — SETUP","peak":"ACT II — PEAK","reflection":"ACT III — REFLECTION"}.get(cur_act,cur_act)
                st.markdown(f'<div style="color:{c};font-weight:700;letter-spacing:2px;text-transform:uppercase;'
                    f'border-left:3px solid {c};padding-left:10px;margin:16px 0 8px">{lbl}</div>',unsafe_allow_html=True)
            st.markdown(f'**{EMO.get(beat.emotion.value,"🎭")} {beat.beat_id}** · _{beat.emotion.value}_ · {fmt(beat.duration_hint_s)}')
            st.markdown(f'> _{beat.narration_text}_')
        sp=result.get("script_json_path")
        if sp and os.path.exists(sp):
            with open(sp) as f: jd=f.read()
            st.download_button("⬇️ Script JSON",jd,"script.json","application/json")

    with t3:
        stage_names={"visual_understanding":"🎞️ Visual","audio_understanding":"🎙️ Transcription",
            "narrative_engine":"📖 Scripts","voice_synthesis":"🔊 Narration",
            "music_generation":"🎵 Music","ambient_sound":"🌊 Ambient","video_assembly":"🎬 Assembly"}
        for k,lbl in stage_names.items():
            t=timings.get(k,0)
            a,b,cc=st.columns([2,3,1])
            with a: st.markdown(lbl)
            with b: st.progress(min(t/max(tot,1),1.0))
            with cc: st.markdown(f"`{t:.1f}s`")
        st.markdown(f"**Total: `{tot:.1f}s`**")

    st.divider()
    ca,_,cb=st.columns([1,2,1])
    with ca:
        if st.button("🔄 Back to Customize",use_container_width=True):
            st.session_state.stage="editing"; st.rerun()
    with cb:
        if st.button("🆕 New Documentary",use_container_width=True):
            for k in list(DEFS.keys()): st.session_state.pop(k,None)
            st.rerun()
