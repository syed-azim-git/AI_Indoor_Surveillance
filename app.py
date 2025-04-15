import streamlit as st
import os
import subprocess
import ast
import re 

# Set Streamlit layout
st.set_page_config(layout="wide")
st.markdown("## AI-Driven Context Detection for Comprehensive Surveillance")

# Acknowledgments
with st.expander("📌 Acknowledgments", expanded=True):
    st.markdown("""
**Final Year Project by**  
Shreyas Sai R (3122213002096)  
Syed Azim (3122213002110)  

**Mentored by**  
Dr. P. Vijayalakshmi  

**Guided by**  
Dr. N. Venkateswaran  
Dr. S. Karthie  
""")

# Layout
left_col, right_col = st.columns([5, 7])
with left_col:
    st.subheader("📦 Logs")
with right_col:
    st.subheader("🧠 LLM Description")

# Critical activity keywords
critical_keywords = {
    'Violence', 'arguing', 'crying', 'punching person (boxing)', 'slapping',
    'smashing', 'throwing tantrum', 'wrestling',
    'Medical Emergency', 'bandaging', 'using inhaler',
    'Fire Emergency', 'extinguishing fire', 'lighting fire',
    'Theft', 'lock picking', 'opening door', 'packing',
    'Smoking', 'smoking', 'smoking hookah', 'smoking pipe'
}

# Cumulative logs
combined_logs = ""
combined_llm = ""
alert_mode = False

for i in range(1, 12):
    # === SCP Transfer ===
    scp_command = f"scp -P 11114 C:/Users/AZIMAISHA/Downloads/Features/feature_{i}.h5 root@sshg.jarvislabs.ai:/home/Features"
    os.system(scp_command)

    # === SSH Execution ===
    remote_cmd = (
        f"cd /home && "
        f"python3 -m inference best_model_2_MFCheck.pth hello.csv Features/feature_{i}.h5"
    )
    ssh_command = (
        f"ssh -o StrictHostKeyChecking=no -p 11114 root@sshg.jarvislabs.ai \"{remote_cmd}\""
    )

    full_output = ""
    process = subprocess.Popen(ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    lines = []
    while True:
        line = process.stdout.readline()
        if not line:
            break
        full_output += line
        lines.append(line)

    process.wait()

    parsed_lines = full_output.strip().split('\n')
    try:
        array_output = ast.literal_eval(parsed_lines[0]) if parsed_lines and parsed_lines[0].strip().startswith('[') else []
    except Exception:
        array_output = []
        lines.insert(0, "⚠️ Could not parse array output.")

    llm_description = "\n".join(parsed_lines[1:]) if len(parsed_lines) > 1 else "No Description Output"

    # === Check for critical activity ===
    current_alert = any(item in critical_keywords for item in array_output)
    action_msg = ""

    if current_alert or alert_mode:
        alert_mode = True  # once alert is on, it persists

        if current_alert:
            user_response = st.radio(
                f"⚠️ Critical activity detected in feature_{i}.h5! Confirm alert?",
                ["True Alert", "False Alert"],
                key=f"alert_{i}"
            )

            if user_response == "False Alert":
                alert_mode = False
                action_msg = ("🌿 Nothing critical.\n"
                              "❌ Feature file and video deleted.")
            else:
                action_msg = ("🚨 System on alert.\n"
                              "Informed to administrator 👨🏻‍💼.\n"
                              "Videos from this instant will be recorded and saved 💾.\n"
                              "🎥 Live stream to administrator turned on.")
        else:
            # Already in alert mode
            action_msg = ("🚨 System on alert (continued).\n"
                          "💾 Video saved for reference.")
    else:
        action_msg = ("🌿 Nothing critical.\n" 
                      "❌ Feature file and video deleted.")

    # Combine log
    log_text = f"""🎥 Video saved as video_{i}
Feature extracted & saved as feature_{i}.h5 📄 
📤 Transferring feature_{i}.h5 to Cloud ☁️...
feature_{i}.h5 Successfully Transferred ✅

🧮 Array output:\n {array_output}

{action_msg}
"""
    llm_text = f"""
🧮 Array output: {array_output}

{re.sub(r'[\n]', ' ', action_msg, count=1)}
{llm_description}
"""
    combined_logs += log_text + "\n"
    combined_llm += f"LLM Description for video_{i}.h5:\n{llm_text}\n\n"

    # === UI Update per Iteration ===
    with st.container():
        left_col, right_col = st.columns([5, 7])
        with left_col:
            st.code(log_text, language="text")
        with right_col:
            st.code(llm_text, language="text")

# === Download section ===
st.markdown("---")
st.subheader("📥 Download Full Logs")

output_txt = f"{combined_logs}\n{'='*30}\n{combined_llm}"

st.download_button(
    label="📄 Download Combined Logs & LLM Output",
    data=output_txt,
    file_name="inference_logs_llm.txt",
    mime="text/plain"
)
