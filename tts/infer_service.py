from flask import Flask, request, jsonify
import os
import gc
import time
import logging
import hashlib
import torch
import traceback
from tts.infer_cli import MegaTTS3DiTInfer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("infer_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量，存储已初始化的实例
global infer_ins
infer_ins = None

def get_infer_instance():
    """懒加载方式获取推理实例"""
    global infer_ins
    if infer_ins is None:
        logger.info("初始化 MegaTTS3DiTInfer 实例...")
        infer_ins = MegaTTS3DiTInfer()
        logger.info("MegaTTS3DiTInfer 实例初始化完成")
    return infer_ins

@app.route('/process', methods=['POST'])
def process():
    # 获取请求参数
    data = request.json
    wav_path = data.get('wav_path')
    input_text = data.get('input_text')
    output_dir = data.get('output_dir', './output')
    time_step = data.get('time_step', 32)
    p_w = data.get('p_w', 1.6)
    t_w = data.get('t_w', 2.5)

    if not wav_path or not input_text:
        return jsonify({'error': '缺少必要参数: wav_path 或 input_text'}), 400

    # 检查文件是否存在
    if not os.path.exists(wav_path):
        return jsonify({'error': f'文件不存在: {wav_path}'}), 404

    # 检查latent文件是否存在
    latent_file = wav_path.replace('.wav', '.npy')
    if not os.path.exists(latent_file):
        logger.warning(f"Latent文件不存在: {latent_file}")
        latent_file = None

    start_time = time.time()
    logger.info(f"开始处理请求: wav_path={wav_path}, text='{input_text}'")

    try:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 获取推理实例
        infer_instance = get_infer_instance()
        
        with open(wav_path, 'rb') as file:
            file_content = file.read()

        # 预处理
        logger.info("开始预处理...")
        resource_context = infer_instance.preprocess(file_content, latent_file=latent_file)
        
        # 推理
        logger.info(f"开始推理: time_step={time_step}, p_w={p_w}, t_w={t_w}")
        wav_bytes = infer_instance.forward(
            resource_context, 
            input_text, 
            time_step=time_step, 
            p_w=p_w, 
            t_w=t_w
        )

        # 保存结果文件
        os.makedirs(output_dir, exist_ok=True)
        filename = hashlib.md5(input_text.encode()).hexdigest()
        output_file = os.path.join(output_dir, f'[P]{filename}.wav')
        
        with open(output_file, 'wb') as f:
            f.write(wav_bytes)
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        # 计算处理耗时
        process_time = time.time() - start_time
        logger.info(f"处理完成: 输出文件={output_file}, 耗时={process_time:.2f}秒")

        return jsonify({
            'output_file': output_file,
            'input_text': input_text,
            'process_time': f"{process_time:.2f}秒"
        })

    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"处理失败: {error_msg}\n{stack_trace}")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        return jsonify({
            'error': error_msg,
            'stack_trace': stack_trace
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({'status': 'ok', 'cuda_available': torch.cuda.is_available()})

if __name__ == '__main__':
    # 预先初始化模型，避免首次请求延迟
    get_infer_instance()
    logger.info("服务已启动，监听 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)