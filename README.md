# Vocalnet(In progress)

基于Wavenet与Waveglow的中文歌声合成系统

## 安装

### 使用Docker（推荐）


### 使用源码
   1. 从Github下载代码

    ```
        git clone https://github.com/xushengyuan/Vocalnet.git
    ```

   2. 安装相关依赖包

    ```
        pip install -r requirements.txt
    ```

   3. 从Github的release中下载预训练模型，保存到项目根目录。大陆用户可使用多线程下载工具（如`aria`）提高下载速度及稳定性。

## 生成
程序接收由Vocalloid编辑器保存的`.vsqx`工程文件，仅读取第一条轨道中的音符。

### 一键合成
    ```
    python Vocalnet_synthesis.py in.vsqx
    ```
### 分布合成
   1. 在vsqx中保存工程到根目录，执行`vsqx2npy.py`将`.vsqx`解析
      
       ```
       python vsqx2npy.py in.vsqx out.npy
       ```

   2. 合成梅尔谱
      
      ```
      python synthesis.py checkpoint_step000800000_ema.pth ./out --conditional="out.npy"
      ```

   3. 合成波形

      ```
          python waveglow_generate.py --checkpoint="model.ckpt-660000.pt" --local_condition_file="./out/checkpoint_step000800000_ema.npy"
      ```

波形文件保存在`checkpoint_step000800000_ema.wav`中

## 训练

### Wavenet


## 演示视频

## 参考

- https://github.com/npuichigo/waveglow
- https://github.com/r9y9/wavenet_vocoder
