使用swin-transformer辨識農作物


環境部分: (參考以下網址https://blog.csdn.net/qq_36622589/article/details/117913064) </br>
1.conda create -n swin python=3.7 -y </br>
2.activate swin </br>
3.conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch </br>
4.pip install timm==0.3.2 </br>
5.pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 </br> 
6.去此網址https://github.com/NVIDIA/apex 下載到本地資料夾，使用CD 進去該資料夾底下 </br>
7.python setup.py install 等待安裝 </br>
8.pip install json </br>

預訓練model: https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth </br>

</br>
執行程式碼的事前準備: </br>
1.將原始的訓練資料放入名為train的資料夾，如圖: </br>

![image](https://user-images.githubusercontent.com/103729404/169832361-f835b795-85a5-4b3a-94f0-b3c82eeef7eb.png)

2.在此資料夾新增名為result的資料夾</br>


3.在此資料夾新增名為test的資料夾，裡面是原始的圖片，如圖:</br>

![image](https://user-images.githubusercontent.com/103729404/169832433-946b0817-6036-4d4b-b937-4bf7e2d8b58b.png)

最後所以此資料夾下會有以下的檔案:</br>

![image](https://user-images.githubusercontent.com/103729404/169832291-5778f43c-6f0e-48c4-a906-60e66e2d3b84.png)


code 執行流程:</br>
  1. train.py (開始訓練model) </br>
  2. predict.py (輸出結果為result.csv) </br>




  
