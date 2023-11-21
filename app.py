import streamlit as st
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import cv2

model = torch.load('/kaggle/input/pytorch-model/full_model.pth', map_location=torch.device('cpu'))
names = ['No DR','Mild','Moderate','Severe','Proliferative DR']

st.title('Diabetic Retinopathy Detection')
st.divider()

on = st.toggle('Single eye')

def crop_image_from_gray(img,tol=7):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray_img>tol

    check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
    img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
    img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
    img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    img = np.stack([img1,img2,img3],axis=-1)
    return img

val_transforms = A.Compose(
    [
        A.Resize(height=728, width=728),
        A.Normalize(
            mean=[0.3199, 0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

def make_prediction(model, img):
    x = val_transforms(image=img)["image"].unsqueeze(0)
    
    model.eval()

    with torch.no_grad():
        pred = model(x)
        pred[pred < 0.75] = 0
        pred[(pred >= 0.75) & (pred < 1.5)] = 1
        pred[(pred >= 1.5) & (pred < 2.4)] = 2
        pred[(pred >= 2.4) & (pred < 3.4)] = 3
        pred[pred >= 3.4] = 4
        pred = pred.long().squeeze(1)
        
    model.train()
    return pred.cpu().numpy()[0]

if on:
    u_img = st.file_uploader('Left or right eye', type=['jpg', 'jpeg', 'png'], key=0)


    if u_img is not None:
        img = Image.open(u_img)
        st.image(img.resize((1024,1024)), width=400)

        image = crop_image_from_gray(np.array(img))

        if st.button('Predict', use_container_width=True, key=4):
            result = make_prediction(model, image)

            st.divider()
            img_cols = st.columns(2)
            img_cols[0].header(f'Result: {names[result]}')

else:
    img_cols = st.columns(2)
    right = img_cols[0].file_uploader('Left eye', type=['jpg', 'jpeg', 'png'], key=1)
    left = img_cols[1].file_uploader('Right eye', type=['jpg', 'jpeg', 'png'], key=2)

    if right is not None and left is not None:
        right_img = Image.open(right)
        img_cols[0].image(right_img.resize((1024,1024)), width=300)

        left_img = Image.open(left)
        img_cols[1].image(left_img.resize((1024,1024)), width=300)

        right_image = crop_image_from_gray(np.array(right_img))
        left_image = crop_image_from_gray(np.array(left_img))

        if st.button('Predict', use_container_width=True, key=3):

            img_cols[0].divider()
            img_cols[1].divider()

            result0 = make_prediction(model, right_image)
            img_cols[0].header(f'Result: {names[result0]}')

            result1 = make_prediction(model, left_image)
            img_cols[1].header(f'Result: {names[result1]}')
