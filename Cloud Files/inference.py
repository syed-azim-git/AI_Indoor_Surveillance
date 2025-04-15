import torch
import numpy as np
from model import AudioVisualFusionModel
import h5py
import pandas as pd
import sys
import ollama

if len(sys.argv) != 4:
    print("Usage: python3 -m inference <path_to_your_model_state_dict> <path_to_map_csv> <path_to_features>")
    sys.exit(1)

path1 = sys.argv[1]
path2 = sys.argv[2]
path3= sys.argv[3]

def inference(model, audio_np, visual_np, device):
 try:
    model.eval()

    # For model Sake
    if audio_np.ndim == 2:
        audio_np = np.expand_dims(audio_np, axis=0)
    if visual_np.ndim == 2:
        visual_np = np.expand_dims(visual_np, axis=0)

    # Convert to torch tensor 
    audio_tensor = torch.tensor(audio_np, dtype=torch.float32).to(device)
    visual_tensor = torch.tensor(visual_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(audio_tensor, visual_tensor)
        probs = torch.softmax(output, dim=2)
        preds = torch.argmax(probs, dim=2)

    return preds.cpu().numpy(), probs.cpu().numpy()
 except Exception as e:
    print(f"Error during inference: {str(e)}")
    return None, None

model = AudioVisualFusionModel(
    audio_dim=40,
    visual_dim=1024,
    hidden_dim=256,
    fusion_dim=512,
    num_categories=112
)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

model=model.to(device)

model.load_state_dict(torch.load(path1, map_location=torch.device('cpu')))

with h5py.File(path3, 'r') as f:
      visual_features=np.array(f['visual_features'])
      audio_features=np.array(f['audio_features'])

a,b=inference(model,audio_features,visual_features,device)

def load_class_mapping(csv_path):
    mapping_df = pd.read_csv(csv_path)
    # Assuming first column is number and second is class name
    class_mapping = dict(zip(mapping_df.iloc[:, 1], mapping_df.iloc[:, 0]))
    return class_mapping

class_mapping=load_class_mapping(path2)
l=[]
arr=a+1
for i in range (5):
    keys_list = [k for v in arr for k, val in class_mapping.items() if val == v[i].item()]
    l.append(keys_list[0])
#print(l) 

# llama3.2:1b   
model = "llama3.2:1b"

# System Prompt 
system_prompt = '''Here are few examples on how you have to give your answers: 
### Example 1  
**Input Classes:**  
shouting, talking on the phone, running, falling, extinguishing fire  

**Generated Context:**  
"Possible emergency situation. Someone might be calling for help while another person is running, possibly evacuating. Firefighting actions suggest an ongoing fire incident."

### Example 2  
**Input Classes:**  
arguing, slapping, crying, falling, using inhaler  

**Generated Context:**  
"Escalating conflict. A physical altercation may have occurred, leading to distress and injury. The use of an inhaler suggests potential breathing difficulty, possibly due to stress or an asthma attack."

### Example 3  
**Input Classes:**  
waiting in line, using phone, yawning, reading book, watching TV  

**Generated Context:**  
"Casual waiting scenario. People are engaged in passive activities like reading, using their phones, and watching TV while waiting in a queue."

### Example 4  
**Input Classes:**  
packing, opening door, lock picking, looking at phone, walking  

**Generated Context:**  
"Possible security concern. Someone might be unlocking a door or picking a lock while another person appears distracted by their phone. Packing suggests movement, potentially leaving or entering a location."
'''


user_question = f'''
**Input Classes:**  
<{l}>  

**Generated Context:**  
"<The model should infer the likely situation based on the combination of events.> Keep the answer shot, no need explaination, just one small context sentence"

'''

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_question
    }
]

response = ollama.chat(
    model=model,
    messages=messages
)

result = response["message"]["content"]
print(l)
print(result)
