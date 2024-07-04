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
import argparse

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
    
    # 平滑梯度的平方
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

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./TransUNet/prediction/inference')
args = parser.parse_args()

if __name__ == "__main__":
    # load data
    train_dir = args.root_path
    image_list = sorted(os.listdir(os.path.join(train_dir,'image')))
    label_list = sorted(os.listdir(os.path.join(train_dir,'pred')))

    for idx in range(len(image_list)):
        image_file = image_list[idx]
        image_path = os.path.join(train_dir,'image',image_file)

        label_file = label_list[idx]
        label_path = os.path.join(train_dir,'pred',label_file)

        image = tf.imread(image_path)/65535. # 16bit
        image = (image - 0.094) / 0.049 # std = 0.049, mean = 0.094
        label = tf.imread(label_path) 
        label = label.astype(np.uint8) > 0

        logging.basicConfig(filename=f'./RCA/model/{image_file.split(".")[0]}_log.txt', level=logging.INFO,
                            format='%(asctime)s - %(message)s')

        logging.info(f'Training {image_file.split(".")[0]}!')

        # feature construction
        features = []
        feature_names = {}
        sigmas = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10]

        d,h,w = image.shape

        for z in range(d):
            image_d = image[z,:,:]
            silce_features = []
            for i in range(len(sigmas)):
                if i == 0:
                    gs = gaussion_smoothing(image_d,sigma=sigmas[0])
                    feature_names[f'gs_{i}'] = gs.flatten()
                else:
                    sigma1 = sigmas[i-1]
                    sigma2 = sigmas[i]

                    gs = gaussion_smoothing(image_d,sigma=sigma2)
                    log = laplace_of_gaussion(image_d,sigma=sigma2)
                    ggm = gaussion_gradient_magnitude(image_d,sigma=sigma2)
                    dog = difference_of_gaussion(image_d,sigma1=sigma1,sigma2=sigma2)
                    s_lambda1,s_lambda2 = structure_tensor_eigenvalue(image_d,sigma=sigma2)
                    h_lambda1,h_lambda2 = hessian_of_guassion_eigenvalue(image_d,sigma=sigma2)

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
                silce_features.append(value)

            silce_features = np.stack(silce_features,axis=0)
            silce_features = silce_features.transpose(1,0)
            features.append(silce_features)

        # Set a reasonable train_num and make trade-offs between precision and time
        train_num = 35000
        features = np.stack(features,axis=0)
        num_feat = features.shape[-1]
        features = features.reshape(-1,num_feat)[:train_num,:]
        label = label.flatten()[:train_num]


        logging.info(f'Features loaded: {len(label):d} samples and {features.shape[1]:d} features totally')

        # split train and test dataset
        X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=42)

        # train random_forest_model
        rf_classifier = RandomForestClassifier(n_estimators=100, max_features=0.7, random_state=42)
        logging.info(f'Forest ready: {rf_classifier.n_estimators:d} trees')

        start = time.time()
        rf_classifier.fit(X_train, y_train)
        end = time.time()
        train_time = end - start
        logging.info(f'Train time:{train_time}s')

        # save random_forest_model
        joblib.dump(rf_classifier, f'./RCA/model/random_forest_model_{image_file.split(".")[0]}.pkl')
        logging.info(f'Forest saved!')

        # prediction and evaluate
        start = time.time()
        y_pred = rf_classifier.predict(X_test)
        end = time.time()
        test_time = end - start
        logging.info(f'Test time:{test_time}s')

        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Accuracy: {accuracy:.2f}")

        logging.info("Classification Report:")
        logging.info(classification_report(y_test, y_pred))
        
        feature_importances = rf_classifier.feature_importances_
        for (feature, _), importance in zip(feature_names.items(), feature_importances):
            logging.info(f"{feature}: {importance:.4f}")