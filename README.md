[Japanese/English]

---
# NARUTO-HandSignDetection
物体検出を用いてNARUTOの印を検出するモデルとサンプルプログラムです。
<!--
![header](https://user-images.githubusercontent.com/37477845/95489808-4fb55c80-09d2-11eb-95f0-c3cdc6d55d83.png)
-->
<div align="left">

<img src="https://user-images.githubusercontent.com/37477845/95489944-78d5ed00-09d2-11eb-96f6-a687b012c413.gif" width="45%">　<img src="https://user-images.githubusercontent.com/37477845/95645297-97360880-0af8-11eb-9134-d92cbfb5fe42.gif" width="40%"><!--　<img src="https://user-images.githubusercontent.com/37477845/95490010-93a86180-09d2-11eb-8185-e50fd2b5c137.gif" width="45%">--><br>

右図：© NARUTO -ナルト- 9話『写輪眼のカカシ』岸本斉史作/集英社/studioぴえろ<br>
※<span id="cite_ref-1">日本の著作権法 第二十条「同一性保持権」</span><sup>[1](#cite_note-1)</sup>における「改変」に相当する可能性があるため、<br>
　右図のバウンディングボックスのオーバーレイ表示は行わないようにしています。

<!-- ![21](https://user-images.githubusercontent.com/37477845/95489944-78d5ed00-09d2-11eb-96f6-a687b012c413.gif)![22](https://user-images.githubusercontent.com/37477845/95490010-93a86180-09d2-11eb-8185-e50fd2b5c137.gif) -->
</div>
<!--
![footer](https://user-images.githubusercontent.com/37477845/95489817-5348e380-09d2-11eb-9df0-3ddd06703c55.png)
-->

---

# Title
Deep写輪眼：オブジェクト検出 EfficientDet を用いた NARUTO の印認識システムの開発<br>
Deep写輪眼：Development of NARUTO's Hand Sign Recognition System using Object Detection EfficientDet

# Abstract
このリポジトリは、<span id="cite_ref-2">NARUTO</span><sup>[2](#cite_note-2)</sup> の印を認識するための訓練済みモデルとサンプルプログラムを公開しています。<br><br>
忍術の発動は、一部の忍術をのぞき手で印を結ぶことが必要です。<br>
また、性質変化は印に特徴が現れるため(火遁→寅の印、土遁→亥の印など)、<br>
印を素早く認識することが出来れば、忍同士の戦闘においてアドバンテージを得ることが出来ます。<br>
印の認識にはディープラーニングの物体検出モデルの一つEfficientDet-D0を使用することで、<br>
過去に試験的に作成していたDeep写輪眼(2019年版v1)よりも精度を大幅にアップしました。<br>
(Tensorflow2 Object Detection APIを使用)

In this repository, we discuss the <span id="cite_ref-2">NARUTO's</span><sup>[2](#cite_note-2)</sup> hand sign recognition system that we have developed. <br>

<!--# Introduction
-->
# Requirements
* Tensorflow 2.3.0 or Later
* OpenCV 3.4.2 or Later
* Pillow 6.1.0 or Later

# DataSet
### データセットについて
データセットは非公開です（訓練済みのモデルは公開します）<br>
※<span id="cite_ref-3">日本の著作権法 第四十七条の七「複製権の制限により作成された複製物の譲渡」</span><sup>[1](#cite_note-1)</sup>に準拠

また、自分で撮影した画像、アニメ画像の他に、[naruto-hand-sign-dataset](https://www.kaggle.com/vikranthkanumuru/naruto-hand-sign-dataset)を利用しています。

### 印の種類
14種類(子～亥、壬、合掌)の印に対応しています。<br>
|子(Ne/Rat)|丑(Ushi/Ox)|寅(Tora/Tiger)|卯(U/Hare)| 
|:---:|:---:|:---:|:---:| 
|<img src="https://user-images.githubusercontent.com/37477845/95611897-6d032d00-0a9d-11eb-86c4-de1c50c0d7b6.jpg" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95611906-6ffe1d80-0a9d-11eb-9054-4e68c42e52ca.jpg" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95611912-712f4a80-0a9d-11eb-8cb8-fc7097e16f60.jpg" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95611915-72607780-0a9d-11eb-9995-66524ce4f978.jpg" width="100%">| 

|辰(Tatsu/Dragon)|巳(Mi/Snake)|午(Uma/Horse)|未(Hitsuji/Ram)| 
|:---:|:---:|:---:|:---:| 
|<img src="https://user-images.githubusercontent.com/37477845/95611920-7391a480-0a9d-11eb-8e74-db39acf90f83.jpg" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95611922-742a3b00-0a9d-11eb-8a21-8bdf207db9bb.jpg" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95611928-755b6800-0a9d-11eb-86c0-67605ffd6e9b.jpg" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95611930-768c9500-0a9d-11eb-81c6-067b632dc43d.jpg" width="100%">| 

|申(Saru/Monkey)|酉(Tori/Bird)|戌(Inu/Dog)|亥(I/Boar)| 
|:---:|:---:|:---:|:---:| 
|<img src="https://user-images.githubusercontent.com/37477845/95611931-77252b80-0a9d-11eb-97d6-e3efc6f1aac3.jpg" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95611935-77bdc200-0a9d-11eb-95e1-feb8bf7f61de.jpg" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95611936-78eeef00-0a9d-11eb-90b3-f565e4763c50.jpg" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95611938-7a201c00-0a9d-11eb-9d5f-1daf2405f20f.jpg" width="100%">| 

|壬(Mizunoe)|合掌(Gassho/Hnad Claps)|-|-|
|:---:|:---:|:---:|:---:|
|<img src="https://user-images.githubusercontent.com/37477845/95611947-7c827600-0a9d-11eb-97ae-9d7eabc58cd5.jpg" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95611943-7b514900-0a9d-11eb-97be-4fda80d17879.jpg" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95613470-f61b6380-0a9f-11eb-8c75-e6efce443d3a.jpg" width="50%">|<img src="https://user-images.githubusercontent.com/37477845/95613470-f61b6380-0a9f-11eb-8c75-e6efce443d3a.jpg" width="50%">|

### データセットの枚数
総枚数：6377枚(内アニメ画像：2651枚)<br>
タグ付き枚数：4903枚<br>
タグ無し枚数：1474枚<br>
アノテーションボックス数：6037個<br>
<img src="https://user-images.githubusercontent.com/37477845/95611949-7db3a300-0a9d-11eb-97a9-dc988bd3f608.png" width="35%">　<img src="https://user-images.githubusercontent.com/37477845/95611950-7e4c3980-0a9d-11eb-9bcb-72888a9aaebb.png" width="50%">

# Trained Model

# Results

<!-- # Discussion(考察)
-->
<!-- # Conclusions(結論)
-->
# Usage

# Application Example(応用例)
<!--
* スマートグラス
* 認証 「****」開錠
* アカデミー卒業試験対策 〇〇×
* リモートコントロール
-->

# Acknowledgements(謝辞)

# References
1. [^](#cite_ref-1)<span id="cite_note-1">日本：[著作権法 第二十条「同一性保持権」](https://elaws.e-gov.go.jp/search/elawsSearch/elaws_search/lsg0500/detail?lawId=345AC0000000048#183)</span>
1. [^](#cite_ref-2)<span id="cite_note-2">岸本斉史作『[NARUTO](https://www.shonenjump.com/j/rensai/naruto.html)』集英社、1999年-2014年</span>
1. [^](#cite_ref-3)<span id="cite_note-3">日本：[著作権法 四十七条の七「複製権の制限により作成された複製物の譲渡」](https://elaws.e-gov.go.jp/search/elawsSearch/elaws_search/lsg0500/detail?lawId=345AC0000000048#407)</span>
1. ^<span id="cite_note-4">XXXX</span>
1. ^<span id="cite_note-5">XXXX</span>

# Authors
高橋かずひと(https://twitter.com/KzhtTkhs)
<!--
# Affiliations(所属)
-->

# License 
NARUTO-HandSignDetection is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

# License(Font)
衡山毛筆フォント(https://opentype.jp/kouzanmouhitufont.htm)
