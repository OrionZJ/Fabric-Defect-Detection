import cv2
import numpy as np 
from numpy.linalg import norm, svd

def inexact_augmented_lagrange_multiplier(X, lmbda = 0.01, tol = 1e-7, maxIter = 1000):
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y /dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n= Y.shape[1]
    itr = 0
    while True:
        Eraw = X - A + (1/mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
        U, S, V = svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(0.05 * n), n])

        Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        A = Aupdate
        E = Eupdate
        # print(itr)
        Z = X - A - E
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / dnorm) < tol) or (itr >= maxIter):
            break
    print("IALM Finished at iteration %d" % (itr))
    return A, E

if __name__ == "__main__":
    imgfile = "./1a7d23bee7a332c01403047173.jpg"
    imgfile = "./1a8f82fe1d53bfc90939515204.jpg"
    import cv2
    from matplotlib import pyplot as plt
    scale = 0.1
    # 读入jpg图片
    img = cv2.imread(imgfile)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    
    # 将图片转化为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    import time
    t1 = time.perf_counter()
    A, E = inexact_augmented_lagrange_multiplier(gray_img, maxIter=30)
    print(time.perf_counter() - t1)

    # 将灰度图转化为array
    img_array = np.array(E, dtype=np.uint8)


    # 将array转化回图片
    array_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    # 放大回原来的大小
    array_img = cv2.resize(array_img, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)

    # 显示图片
    plt.imshow(array_img)
    plt.axis('off')
    plt.show()





