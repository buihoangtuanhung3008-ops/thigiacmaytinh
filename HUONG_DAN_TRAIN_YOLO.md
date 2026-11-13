# Huong dan train YOLO nhan dien vat can

## 1. Hieu dung ve weights

- `yolo11n.pt` la weights YOLO da duoc train san tren COCO. No chay nhan dien thu ngay, nhung chua duoc train rieng de phan biet cac vat can trong du lieu cua ban.
- Khong mo/sua `.pt` bang VS Code nhu file van ban. Day la file nhi phan.
- Sau khi train, file can dung trong ung dung la `runs/detect/vat_can/weights/best.pt`. Giữ `yolo11n.pt` lam weights goc de co the train lai.
- Anh/video chua gan nhan khong the dung de train supervised. Can ve khung bao (bounding box) cho tung vat can.

## 2. Du lieu hien co

Dataset Roboflow trong `anh yolo/Warehouse Robot Docking and Path.v2-vatcanv1.yolov11 (1)` hien co 5 anh train va 1 anh validation. Dataset co hai nhan `san` va `vat_can`; script train ben duoi tu dong bo nhan `san`, chuyen cac polygon vat can thanh bounding box, va train model mot lop `vat_can`.

Sieu it anh, dac biet validation chi co 1 anh, nen model chi dung de thu nghiem. Hay bo sung them anh co vat can o nhieu kich thuoc, goc nhin, dieu kien sang va khung canh khac nhau; tach cac canh quay rieng giua train va validation.

## 3. Gan nhan

Dung mot cong cu gan nhan co the xuat dinh dang **Ultralytics YOLO Detection** (vi du CVAT chay local). Tao project/task, khai bao class `vat_can`, import anh va video, ve bounding box sat quanh phan vat nhin thay duoc, roi export dinh dang YOLO Detection.

Nen chon khung video cach nhau khoang 0.5-1 giay, bo qua nhieu khung lien tiep gan nhu giong nhau. Dua ca truong hop vat gan/xa, nho/lon, sang/toi va cung mau san vao tap nhan. Kiem tra lai label truoc khi export: moi anh co vat can phai co file `.txt` tuong ung; anh khong co vat can co the co file `.txt` rong.

Khong chia ngau nhien cac khung sat nhau vao train va val. Hay tach theo anh/video/canh quay: train khoang 80%, val khoang 20%. Neu chi co mot video, chon cac doan thoi gian tach biet; tranh de cac khung lien tiep cua cung mot vat xuat hien o ca hai tap.

## 4. Train tren Windows

Mo PowerShell tai thu muc project, dung Python environment da cai cac package trong `requirements.txt`. Kiem tra duong dan anh va label truoc:

```powershell
python .\train_yolo.py --check-only
```

Train model YOLO mot lop `vat_can`:

```powershell
python .\train_yolo.py --epochs 50 --batch 2 --imgsz 640
```

Mac dinh script dung `yolo11n.pt`, tu chon CUDA neu PyTorch nhan GPU; neu khong thi chay CPU. Co the chi dinh `--device cpu` hoac `--device 0`. Tren Windows script dat `workers=0` de tranh loi tao process.

Script sua duong dan YAML Roboflow bi lech bang cach tao YAML tam, khong sua dataset goc. Anh va label cung duoc chuyen tam trong luc train; chi class `vat_can` duoc giu va remap thanh class 0, cac label san bi loai bo. Ket qua train duoc luu trong:

```text
runs/detect/vat_can/weights/best.pt
runs/detect/vat_can/weights/last.pt
```

Neu ten `vat_can` da ton tai, Ultralytics tu tang so thu tu ten run. Dung `best.pt` de nhan dien; `last.pt` la weights o epoch cuoi, khong nhat thiet tot nhat.

### Train segmentation de to mask san va vat can

Label Roboflow goc co polygon cho ca `san` va `vat_can`. Script detector ben tren chi giu class `vat_can` va chuyen polygon thanh box. De tao model segmentation giu nguyen ca hai class va polygon:

```powershell
python .\train_yolo_seg.py --check-only
python .\train_yolo_seg.py --epochs 100 --batch 2 --imgsz 640
```

Script dung `yolo11n-seg.pt` lam pretrained weights, giu nguyen label polygon, va luu model tai `runs/segment/vat_can_seg/weights/best.pt` (Ultralytics se them hau to neu ten da ton tai). Lan train dau co the can tai weights pretrained. Dataset hien tai chi co 5 anh train va 1 anh validation; model segmentation nay chi la demo cho canh tuong tu, chua the khang dinh tong quat sang camera/canh quay khac. Nen bo sung nhieu video/canh quay va tach validation theo video truoc khi tin dung ket qua.

Ban demo da train 50 epoch tren CPU duoc luu tai `runs/segment/floor_seg_demo/weights/best.pt`. Ket qua validation tren duy nhat 1 anh: mask mAP50 0.954, mask mAP50-95 0.618; cac chi so nay rat khong on dinh vi tap validation co 1 anh, khong phai cam ket chat luong ngoai canh train.

## 5. Nap weights vao ung dung

Sau khi train detector, chon weights `best.pt` de nhan dien class `vat_can`. De xem ca mask san va vat can, chon weights segmentation tai `runs/segment/.../weights/best.pt`. Bat **Che do phan vung (thu nghiem)** de hien mask mau; tat mode de tro lai khung box nhu truoc. Weights `yolo11n.pt` hoac detector thuong khong co mask nen khong the phan vung san.

Phat hien trong video duoc ghi vao tab **Lich su**; cac track ID khac nhau co khung bao gan nhau trong 5 giay duoc gop de han che ghi lap. Lich su chi ton tai trong phien dang mo va gioi han 200 dong.

Ung dung mac dinh dung confidence `0.15` de uu tien do nhay; co the dieu chinh trong **Cau hinh model / tracker**. Nguong thap hon co the bat duoc du doan yeu hon nhung cung de bao nham. Video duoc xu ly tung frame theo toc do inference de giu lien tuc tracking; giao dien van phan hoi trong luc YOLO dang xu ly, nhung video co the phat cham hon toc do goc neu may khong xu ly kip.

Kiem tra them bang anh/video khong co trong tap train. Neu vat nho/xa van bi bo sot, bo sung va gan nhan them cac truong hop do, sau do train lai tu `yolo11n.pt`.

Chi so tren tap validation khong thay the viec kiem tra bang anh/video thuc te. Model pretrained va model tu train deu co the bo sot; khong dung ket qua thu nghiem nay de dieu khien thiet bi an toan thuc.
