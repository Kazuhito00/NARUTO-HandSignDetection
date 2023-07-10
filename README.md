[Japanese/[English](README_EN.md)]

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
　右図にはバウンディングボックスのオーバーレイ表示は行っておりません。

<!-- ![21](https://user-images.githubusercontent.com/37477845/95489944-78d5ed00-09d2-11eb-96f6-a687b012c413.gif)![22](https://user-images.githubusercontent.com/37477845/95490010-93a86180-09d2-11eb-8185-e50fd2b5c137.gif) -->
</div>
<!--
![footer](https://user-images.githubusercontent.com/37477845/95489817-5348e380-09d2-11eb-9df0-3ddd06703c55.png)
-->

---

# Title
Deep写輪眼：オブジェクト検出 YOLOX を用いた NARUTO の印認識<br>

# Abstract
このリポジトリは、<span id="cite_ref-2">NARUTO</span><sup>[2](#cite_note-2)</sup> の印を認識するための訓練済みモデルとサンプルプログラムを公開しています。<br><br>
忍術の発動は、一部の忍術をのぞき手で印を結ぶことが必要です。<br>
また、性質変化は印に特徴が現れるため(火遁→寅の印、土遁→亥の印など)、<br>
印を素早く認識することが出来れば、忍同士の戦闘においてアドバンテージを得ることが出来ます。<br>
印の認識にはディープラーニングの物体検出モデルの一つYOLOX-Nanoを使用することで、<br>
前回バージョンのDeep写輪眼(EfficientDet-D0利用)よりも推論速度を大幅にアップしました。<br>

<!--# Introduction
-->
# Requirements
* onnxruntime 1.10.0 or Later
* OpenCV 3.4.2 or Later
* Pillow 6.1.0 or Later (Ninjutsu_demo.pyを動かす場合のみ)
* Tensorflow 2.3.0 or Later (SSD、EfficientDetを動かす場合、あるいは後処理をONNXへマージするときのみ)

# DataSet
### データセットについて
データセットは非公開です（訓練済みのモデルは公開します）<br>
※<span id="cite_ref-3">日本の著作権法 第四十七条の七「複製権の制限により作成された複製物の譲渡」</span><sup>[3](#cite_note-3)</sup>に準拠

また、自分で撮影した画像、アニメ画像の他に、<span id="cite_ref-4">naruto-hand-sign-dataset</span><sup>[4](#cite_note-4)</sup>を利用しています。

### お願い事項
データセットはネット上で収集した画像と、自前で撮影した画像で構成されているため、<br>
背景色や服装によっては検出精度が落ちたり、誤検出する可能性があります。<br>
Issueで誤検出した条件を教えていただると助かります。<br>
可能であれば、誤検出する条件の画像(子～亥、壬、合掌)をいただけると大変助かります。<br>
その際、いただいた画像は学習データセットに追加してモデルの再訓練に使用します。

### 印の種類
14種類(子～亥、壬、合掌)の印に対応しています。<br>

<table>
	<tbody>
		<tr>
			<td width="25%">子(Ne/Rat)</td>
			<td width="25%">丑(Ushi/Ox)</td>
			<td width="25%">寅(Tora/Tiger)</td>
			<td width="25%">卯(U/Hare)</td>
		</tr>
		<tr>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611897-6d032d00-0a9d-11eb-86c4-de1c50c0d7b6.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611906-6ffe1d80-0a9d-11eb-9054-4e68c42e52ca.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611912-712f4a80-0a9d-11eb-8cb8-fc7097e16f60.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611915-72607780-0a9d-11eb-9995-66524ce4f978.jpg" width="100%"></td>
		</tr>
	</tbody>
</table>
<table>
	<tbody>
		<tr>
			<td width="25%">辰(Tatsu/Dragon)</td>
			<td width="25%">巳(Mi/Snake)</td>
			<td width="25%">午(Uma/Horse)</td>
			<td width="25%">未(Hitsuji/Ram)</td>
		</tr>
		<tr>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611920-7391a480-0a9d-11eb-8e74-db39acf90f83.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611922-742a3b00-0a9d-11eb-8a21-8bdf207db9bb.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611928-755b6800-0a9d-11eb-86c0-67605ffd6e9b.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611930-768c9500-0a9d-11eb-81c6-067b632dc43d.jpg" width="100%"></td>
		</tr>
	</tbody>
</table>
<table>
	<tbody>
		<tr>
			<td width="25%">申(Saru/Monkey)</td>
			<td width="25%">酉(Tori/Bird)</td>
			<td width="25%">戌(Inu/Dog)</td>
			<td width="25%">亥(I/Boar)</td>
		</tr>
		<tr>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611931-77252b80-0a9d-11eb-97d6-e3efc6f1aac3.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611935-77bdc200-0a9d-11eb-95e1-feb8bf7f61de.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611936-78eeef00-0a9d-11eb-90b3-f565e4763c50.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611938-7a201c00-0a9d-11eb-9d5f-1daf2405f20f.jpg" width="100%"></td>
		</tr>
	</tbody>
</table>
<table>
	<tbody>
		<tr>
			<td width="25%">壬(Mizunoe)</td>
			<td width="25%">合掌(Gassho/Hand Claps)</td>
			<td width="25%">-</td>
			<td width="25%">-</td>
		</tr>
		<tr>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611947-7c827600-0a9d-11eb-97ae-9d7eabc58cd5.jpg" width="100%"></td>
			<td><img src="https://user-images.githubusercontent.com/37477845/95611943-7b514900-0a9d-11eb-97be-4fda80d17879.jpg" width="100%"></td>
			<td></td>
			<td></td>
		</tr>
	</tbody>
</table>

### データセットの枚数
総枚数：10026(内アニメ画像：2651枚)<br>
タグ付き枚数：7098枚<br>
タグ無し枚数：2928枚<br>
アノテーションボックス数：8941個<br>
<img src="https://user-images.githubusercontent.com/37477845/163701957-529d7510-88e4-420f-8099-5dd6f6d9a8cf.png" width="35%">　<img src="https://user-images.githubusercontent.com/37477845/163701962-a4818eba-3b4f-4c92-a7b9-f2331b7f3f23.png" width="50%">

# Trained Model
訓練済みモデルをmodelディレクトリ配下で公開しています。 ※旧バージョンのモデルは「_legacy」ディレクトリに移動
* YOLOX-Nano

# Directory
<pre>
│  simple_demo.py
│  Ninjutsu_demo.py
│
├─model
│  └─yolox
│      │ yolox_nano.onnx
│      └─yolox_onnx.py
│
├─post_process_gen_tools
│      │ convert_script.sh
│      │ make_box_gather_nd.py
│      │ make_boxes_scores.py
│      │ make_cxcywh_y1x1y2x2.py
│      │ make_final_batch_nums_final_class_nums_final_box_nums.py
│      │ make_grids.py
│      │ make_input_output_shape_update.py
│      │ make_nms_outputs_merge.py
│      └─make_score_gather_nd.py
│
├─setting─┬─labels.csv
│         └─jutsu.csv
│
├─utils
│
└─_legacy
</pre>
#### simple_demo.py
　シンプルな検出デモです。<br>
　<img src="https://user-images.githubusercontent.com/37477845/95647513-06b4f380-0b0b-11eb-8caf-5cb092ccdb66.jpg" width="35%">

#### Ninjutsu_demo.py
　忍術判定のデモです。<br>
　印の履歴から術データ(jutsu.csv)にマッチする術名を表示します。<br>
　<img src="https://user-images.githubusercontent.com/37477845/95490010-93a86180-09d2-11eb-8185-e50fd2b5c137.gif" width="35%">
　<!--<img src="https://user-images.githubusercontent.com/37477845/95647523-13394c00-0b0b-11eb-935b-a5a94e2f523d.jpg" width="35%">-->

#### model
　訓練済みモデルを格納しています。

#### post_process_gen_tools
　ONNXにすべての後処理をマージするスクリプト群を格納しています。

#### setting
　ラベルデータ(labels.csv)と術名データ(jutsu.csv)を格納しています。
* labels.csv<br>
印のラベル名を保持しています。<br>
    * A列：英語の印
    * B列：日本語の印
* jutsu.csv<br>
術名と必要印を保持しています。<br>
    * A列：日本語の術種別(火遁等)
    * B列：英語の術種別(火遁等)
    * C列：日本語の術名
    * D列：英語の術名
    * E列以降：術の発動に必要な印

#### utils
　FPS計測用モジュール(cvfpscalc.py)と文字列描画用モジュール(cvdrawtext.py)を格納しています。<br>
　Ninjutsu_demo.pyのみで使用します。

# Usage
デモの実行方法は以下です。
```bash
python simple_demo.py
python simple_demo_without_post.py
python Ninjutsu_demo.py
```

また、デモ実行時には、以下のオプションが指定可能です。
<details>
<summary>オプション指定</summary>

* --device<br>
カメラデバイス番号の指定<br>
デフォルト：
    * simple_demo.py：0
    * Ninjutsu_demo.py：0
* --file<br>
動画ファイル名の指定 ※指定時はカメラデバイスより優先し動画を読み込む<br>
デフォルト：
    * simple_demo.py：None
    * Ninjutsu_demo.py：None
* --fps<br>
処理FPS ※推論時間がFPSを下回る場合のみ有効<br>
デフォルト：
    * simple_demo.py：30
    * Ninjutsu_demo.py：30
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：
    * simple_demo.py：960
    * Ninjutsu_demo.py：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：
    * simple_demo.py：540
    * Ninjutsu_demo.py：540
* --skip_frame<br>
カメラ or 動画読み込み時に何枚おきに処理を実行するか<br>
デフォルト：
    * simple_demo.py：0
    * Ninjutsu_demo.py：0
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：
    * simple_demo.py：model/yolox/yolox_nano.onnx
    * Ninjutsu_demo.py：model/yolox/yolox_nano.onnx
* --input_shape<br>
モデルの入力サイズ<br>
デフォルト：
    * simple_demo.py：416,416
    * Ninjutsu_demo.py：416,416
* --score_th<br>
クラス判別の閾値<br>
デフォルト：
    * simple_demo.py：0.7
    * Ninjutsu_demo.py：0.7
* --nms_th<br>
NMSの閾値<br>
デフォルト：
    * simple_demo.py：0.45
    * Ninjutsu_demo.py：0.45
* --nms_score_th<br>
NMSのスコア閾値<br>
デフォルト：
    * simple_demo.py：0.1
    * Ninjutsu_demo.py：0.1
* --sign_interval<br>
前回の印検出時から指定時間(秒)経過すると印の履歴をクリア<br>
デフォルト：
    * Ninjutsu_demo.py：2.0
* --jutsu_display_time<br>
術成立時に術名を表示する時間(秒)<br>
デフォルト：
    * Ninjutsu_demo.py：5
* --use_display_score<br>
印検出スコアを表示するか否か<br>
デフォルト：
    * Ninjutsu_demo.py：False
* --erase_bbox<br>
バウンディングボックスのオーバーレイ表示を消去するか否か<br>
デフォルト：
    * Ninjutsu_demo.py：False
* --use_jutsu_lang_en<br>
術名表示に英語表記を使用するか否か<br>
デフォルト：
    * Ninjutsu_demo.py：False
* --chattering_check<br>
印を何回連続で検出したら印の成立とみなすか(印の検出チラつき対策)<br>
デフォルト：
    * Ninjutsu_demo.py：1
* --use_fullscreen<br>
フルスクリーン表示を利用するか否か(試験的機能)<br>
デフォルト：
    * Ninjutsu_demo.py：False
</details>

# Application Example
アプリケーションの応用事例を紹介します。
* [第15回UE4ぷちコン「印VADERS」](https://www.youtube.com/watch?v=K4-E5SseVtI)
<!--
|忍認証システム|忍者アカデミー試験対策|Deep写輪眼スマートグラス|
|:---:|:---:|:---:|
|<img src="https://user-images.githubusercontent.com/37477845/95650546-3a9a1400-0b1f-11eb-9b80-c58256b268a3.gif" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95650553-44237c00-0b1f-11eb-8a85-7e5e72e80120.gif" width="100%">|<img src="https://user-images.githubusercontent.com/37477845/95650659-d9267500-0b1f-11eb-90d7-d82cdb2c2824.png" width="100%">|
-->

# Acknowledgements
EfficientDetモデルトレーニング時は、からあげさんの<span id="cite_ref-5">説明記事</span><sup>[5](#cite_note-5)</sup>を参考にいたしました。<br>
また、<span id="cite_ref-6">からあげさんのブログ</span><sup>[6](#cite_note-6)</sup>にて、Deep写輪眼をご紹介いただきました。<br>
大変ありがとうございます。<br><br>
YOLOXのトレーニングには<span id="cite_ref-7">YOLOX-Colaboratory-Training-Sample</span><sup>[7](#cite_note-7)</sup>を使用しています。

# References
1. [^](#cite_ref-1)<span id="cite_note-1">日本：[著作権法 第二十条「同一性保持権」](https://elaws.e-gov.go.jp/search/elawsSearch/elaws_search/lsg0500/detail?lawId=345AC0000000048#183)</span>
1. [^](#cite_ref-2)<span id="cite_note-2">岸本斉史作『[NARUTO](https://www.shonenjump.com/j/rensai/naruto.html)』集英社、1999年-2014年</span>
1. [^](#cite_ref-3)<span id="cite_note-3">日本：[著作権法 四十七条の七「複製権の制限により作成された複製物の譲渡」](https://elaws.e-gov.go.jp/search/elawsSearch/elaws_search/lsg0500/detail?lawId=345AC0000000048#407)</span>
1. [^](#cite_ref-4)<span id="cite_note-4">Kaggle 公開データセット：[naruto-hand-sign-dataset](https://www.kaggle.com/vikranthkanumuru/naruto-hand-sign-dataset)</span>
1. [^](#cite_ref-5)<span id="cite_note-5">[「Object Detection API」で物体検出の自前データを学習する方法（TensorFlow 2.x版）](https://qiita.com/karaage0703/items/8567cc192e151bac3e50)</span>
1. [^](#cite_ref-6)<span id="cite_note-6">からあげさんのブログ：[AIでNARUTO気分！「Deep写輪眼」で遊んでみよう](https://karaage.hatenadiary.jp/entry/2020/10/16/073000)</span>
1. [^](#cite_ref-7)<span id="cite_note-7">[Kazuhito00/YOLOX-Colaboratory-Training-Sample](https://github.com/Kazuhito00/YOLOX-Colaboratory-Training-Sample)</span>

# Authors
高橋かずひと(https://twitter.com/KzhtTkhs)
<!--
# Affiliations(所属)
-->

# License
NARUTO-HandSignDetection is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

# License(Font)
衡山毛筆フォント(https://opentype.jp/kouzanmouhitufont.htm)
