from flask import Flask, request, jsonify
import os
from tts.infer_cli import MegaTTS3DiTInfer

app = Flask(__name__)

# 初始化 MegaTTS3DiTInfer 实例
infer_ins = MegaTTS3DiTInfer()

@app.route('/process', methods=['POST'])
def process():
    # 获取请求参数
    data = request.json
    wav_path = data.get('wav_path')
    input_text = data.get('input_text')
    output_dir = data.get('output_dir', './output')

    if not wav_path or not input_text:
        return jsonify({'error': 'Missing wav_path or input_text'}), 400

    # 处理 WAV 文件
    try:
        with open(wav_path, 'rb') as file:
            file_content = file.read()

        resource_context = infer_ins.preprocess(file_content, latent_file=wav_path.replace('.wav', '.npy'))
        wav_bytes = infer_ins.forward(resource_context, input_text, time_step=32, p_w=1.6, t_w=2.5)

        # 保存结果文件
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'[P]{input_text[:20]}.wav')
        with open(output_file, 'wb') as f:
            f.write(wav_bytes)

        return jsonify({'output_file': output_file})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)