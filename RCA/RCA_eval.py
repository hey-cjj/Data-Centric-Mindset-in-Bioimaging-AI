import numpy as np
import cv2
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import tifffile as tf
from dataset import Vessile
import torch
from tqdm import tqdm
import argparse
from core_select_gpu_K_center import cosine_distance_matrix,prepare_model

def gaussion_smoothing(image,sigma):
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
    return blurred

def laplace_of_gaussion(image,sigma):
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
    log = cv2.Laplacian(blurred, ddepth=cv2.CV_64F,ksize=5)
    return log

def gaussion_gradient_magnitude(image,sigma):
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    ggm = cv2.magnitude(grad_x,grad_y)
    return ggm

def difference_of_gaussion(image,sigma1,sigma2):
    blurred1 = cv2.GaussianBlur(image, (5, 5), sigma1)
    blurred2 = cv2.GaussianBlur(image, (5, 5), sigma2)
    dog = blurred1 - blurred2
    return ggm

def structure_tensor_eigenvalue(image,sigma):
    I_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    I_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    I_x2 = cv2.GaussianBlur(I_x**2, (5, 5), sigma)
    I_y2 = cv2.GaussianBlur(I_y**2, (5, 5), sigma)
    I_xy = cv2.GaussianBlur(I_x*I_y, (5, 5), sigma)

    det = I_x2 * I_y2 - I_xy ** 2
    trace = I_x2 + I_y2

    # Eigenvalues formula for 2x2 matrix
    lambda1 = trace / 2 + np.sqrt((trace / 2) ** 2 - det)
    lambda2 = trace / 2 - np.sqrt((trace / 2) ** 2 - det)

    return lambda1, lambda2

def hessian_of_guassion_eigenvalue(image,sigma):
    image = cv2.GaussianBlur(image, (5, 5), sigma)

    I_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    I_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    I_xx = cv2.Sobel(I_x, cv2.CV_64F, 1, 0, ksize=3)
    I_yy = cv2.Sobel(I_y, cv2.CV_64F, 0, 1, ksize=3)
    I_xy = cv2.Sobel(I_y, cv2.CV_64F, 1, 0, ksize=3)
    
    det = I_xx * I_yy - I_xy ** 2
    trace = I_xx + I_yy

    # Eigenvalues formula for 2x2 matrix
    lambda1 = trace / 2 + np.sqrt((trace / 2) ** 2 - det)
    lambda2 = trace / 2 - np.sqrt((trace / 2) ** 2 - det)

    return lambda1, lambda2

def find_closest_keys(embeddings, dictionary, num=3):
    distances = {key:0 for key,_ in dictionary.items()}
    for embedding in embeddings:
        for key, value in dictionary.items():
            dot_product = np.dot(embedding, value)
            norm_embedding = np.linalg.norm(embedding)
            norm_value = np.linalg.norm(value)
            cosine_similarity = dot_product / (norm_embedding * norm_value)
            cosine_distance = 1 - cosine_similarity
            distances[key] = min(cosine_distance,distances[key])
    
    closest_keys = sorted(distances, key=distances.get)[:num]
    return closest_keys


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./TransUNet/prediction/inference')
parser.add_argument('--save_path', type=str, default='./data/rca')
parser.add_argument('--ckpt', type=str, default='./mae/output/checkpoint-399.pth')
parser.add_argument('--model', type=str, default='mae_vit_base_patch16')
args = parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(filename=f'./RCA/model/log_test.txt', level=logging.INFO,
                        format='%(asctime)s - %(message)s')

    # evaluate dataset(without label)
    src_dir = args.root_path
    src_list = sorted(os.listdir(os.path.join(src_dir,'image')))

    # rca dataset(with label)
    rca_dir = args.save_path

    # load model
    ckpt_dir = args.ckpt
    model = prepare_model(ckpt_dir, args.model)

    # obtain embeddings
    dataset = Vessile(os.path.join(rca_dir,'image'),'test')
    embeddings = {}
    model.eval()
    model.cuda()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,drop_last=False)
    for i, (img, _, name) in tqdm(enumerate(dataloader)):
        embedding, _, _ = model.forward_encoder(img.cuda().float(), mask_ratio=0)
        embeddings[name[0]] = embedding.detach().cpu().numpy().flatten()

    for src_file in src_list:
        image_path = os.path.join(src_dir,'image',src_file)
        image = tf.imread(image_path)/65535. # 16bit
        image = (image - 0.094) / 0.049 # std = 0.049, mean = 0.094
        deep, height, width = image.shape
        stride = 50
        src_embeddings = []
        patch_size = 224
        with torch.no_grad():
            for z in range(deep):
                for y in range(0, height - patch_size + 1, stride):
                    for x in range(0, width - patch_size + 1, stride):
                        patch = torch.from_numpy(image[z,y:y+patch_size, x:x+patch_size])
                        patch = patch.unsqueeze(0).unsqueeze(0)
                        embedding, _, _ = model.forward_encoder(patch.cuda().float(), mask_ratio=0)
                        src_embeddings.append(embedding.detach().cpu().numpy().flatten())

        image_list = find_closest_keys(src_embeddings,embeddings)
        logging.info('Embedding ready')

        # load random forest
        ckpt = f'./RCA/model/random_forest_model_{src_file.split(".")[0]}.pkl'
        loaded_model = joblib.load(ckpt)
        rf_classifier = loaded_model
        logging.info(f'Forest loaded: {ckpt.split("/")[-1]}')
        logging.info(f'Forest ready: {rf_classifier.n_estimators:d} trees')

        dice_max = 0
        dice_avg = 0
        for image_file in image_list:
            image_file = image_file
            image_path = os.path.join(rca_dir,'image',image_file)

            label_file = image_file
            label_path = os.path.join(rca_dir,'label',label_file)

            image = tf.imread(image_path)/65535.
            image = (image - 0.094) / 0.049
            label = tf.imread(label_path)
            label = label.astype(np.uint8)

            # feature construction
            features = []
            feature_names = {}
            sigmas = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10]
            for i in range(len(sigmas)):
                if i == 0:
                    gs = gaussion_smoothing(image,sigma=sigmas[0])
                    feature_names[f'gs_{i}'] = gs.flatten()
                else:
                    sigma1 = sigmas[i-1]
                    sigma2 = sigmas[i]

                    gs = gaussion_smoothing(image,sigma=sigma2)
                    log = laplace_of_gaussion(image,sigma=sigma2)
                    ggm = gaussion_gradient_magnitude(image,sigma=sigma2)
                    dog = difference_of_gaussion(image,sigma1=sigma1,sigma2=sigma2)
                    s_lambda1,s_lambda2 = structure_tensor_eigenvalue(image,sigma=sigma2)
                    h_lambda1,h_lambda2 = hessian_of_guassion_eigenvalue(image,sigma=sigma2)

                    # # feature visualization
                    # cv2.imwrite(f'gs_{i}.png',(gs/gs.max())*255)
                    # cv2.imwrite(f'log_{i}.png',(log/log.max())*255)
                    # cv2.imwrite(f'ggm_{i}.png',(ggm/ggm.max())*255)
                    # cv2.imwrite(f'dog_{i}.png',(dog/dog.max())*255)
                    # cv2.imwrite(f's_lambda1_{i}.png',(s_lambda1/s_lambda1.max())*255)
                    # cv2.imwrite(f's_lambda2_{i}.png',(s_lambda2/s_lambda2.max())*255)
                    # cv2.imwrite(f'h_lambda1_{i}.png',(h_lambda1/h_lambda1.max())*255)
                    # cv2.imwrite(f'h_lambda2_{i}.png',(h_lambda2/h_lambda2.max())*255)
                    
                    feature_names[f'gs_{i}'] = gs.flatten()
                    feature_names[f'log_{i}'] = log.flatten()
                    feature_names[f'ggm_{i}'] = ggm.flatten()
                    feature_names[f'dog_{i}'] = dog.flatten()
                    feature_names[f's_lambda1_{i}'] = s_lambda1.flatten()
                    feature_names[f's_lambda2_{i}'] = s_lambda2.flatten()
                    feature_names[f'h_lambda1_{i}'] = h_lambda1.flatten()
                    feature_names[f'h_lambda2_{i}'] = h_lambda2.flatten()

            for key,value in feature_names.items():
                value = (value-value.mean())/value.std()
                features.append(value)

            features = np.stack(features ,axis=0)
            features = features.transpose(1,0)
            label = label.flatten()

            logging.info(f'Features loaded: {len(label):d} samples and {features.shape[1]:d} features totally')

            # predict
            y_pred = rf_classifier.predict(features)

            # evaluate
            y_pred = y_pred.reshape(224,224)
            label = label.reshape(224,224)
            intersection = np.logical_and(y_pred, label)
            dice = 2. * intersection.sum() / (y_pred.sum() + label.sum())

            y_pred = (y_pred*255).astype(np.uint8)
            save_dir = f'./RCA/prediction/{src_file.split(".")[0]}'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)  
            cv2.imwrite(f"{save_dir}/{image_file.split('.')[0]}.png", y_pred)

            dice_max = max(dice_max,dice)
            dice_avg += dice

            logging.info(f"Dice: {dice:.4f}")

        dice_avg /= len(image_list) 
        logging.info(f"Max Dice: {dice_max:.4f}")
        logging.info(f"Max Dice: {dice_avg:.4f}\n")
