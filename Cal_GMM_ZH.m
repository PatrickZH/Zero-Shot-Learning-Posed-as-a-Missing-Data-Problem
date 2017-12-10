function Px = Cal_GMM_ZH(X,pMiu,pSigma)
%Gaussian posterior probability
%N(x|pMiu,pSigma) = 1/((2pi)^(D/2))*(1/(abs(sigma))^0.5)*exp(-1/2*(x-pMiu)'pSigma^(-1)*(x-pMiu))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% modified by ZH
if size(pSigma,1)==1
    Temp = zeros(size(pSigma,2),size(pSigma,2),size(pSigma,3));
    for i = 1:size(pSigma,3)
        t = diag(pSigma(:,:,i)); 
        Temp(:,:,i) = t;
    end
    pSigma = Temp;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = size(pMiu,1);
[N,D] = size(X);
Px = zeros(N, K);

for i = 1:N
    for k = 1:K
        Px(i,k) = mvnpdf(X(i,:),pMiu(k,:),pSigma(:,:,k));
    end
end

end