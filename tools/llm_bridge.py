import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from http.server import BaseHTTPRequestHandler, HTTPServer

import os

# Create local symlink directory to load the safetensors file instantly
tmp_dir = "/tmp/gemma_local"
os.makedirs(tmp_dir, exist_ok=True)
snapshot_dir = "/Users/kuangtalin/.cache/huggingface/hub/models--unsloth--gemma-2-2b-it/snapshots/"
src_dir = os.path.join(snapshot_dir, os.listdir(snapshot_dir)[0])
for f in os.listdir(src_dir):
    dst = os.path.join(tmp_dir, f)
    if not os.path.exists(dst):
        os.symlink(os.path.join(src_dir, f), dst)

safetensors_src = "/Users/kuangtalin/Documents/google:gemma-4-E2B-it-qat-q4_0-unquantized.safetensors"
safetensors_dst = os.path.join(tmp_dir, "model.safetensors")
if not os.path.exists(safetensors_dst):
    os.symlink(safetensors_src, safetensors_dst)

print("Loading Gemma 2B model from local safetensors...")
model_id = tmp_dir
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    local_files_only=True
).to(device)
print(f"Model loaded successfully on {device}.")

class LlmHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data)
            
            prompts = req.get("prompts", [])
            results = []
            
            # For simplicity, generating sequentially.
            print(f"Received batch of {len(prompts)} prompts.")
            for idx, prompt in enumerate(prompts):
                print(f"Processing {idx+1}/{len(prompts)}...")
                input_text = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                
                # Limit max_new_tokens for speed during testing
                outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
                gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                results.append(gen_text)
                
            res_json = json.dumps({"responses": results})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(res_json.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server_address = ('127.0.0.1', 8081)
    httpd = HTTPServer(server_address, LlmHandler)
    print("LLM Bridge Server running on http://127.0.0.1:8081...")
    httpd.serve_forever()
