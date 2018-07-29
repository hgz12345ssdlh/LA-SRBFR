function [ COEFF, SCORE, LATENT ] = PCA( X )
%PCA: Principle Component Analysis
%    Input: X, each row is a sample, each column is a feature
%    Output: COEFF, principle base vectors of re-based sample space
%            SCORE, re-based X with only principle components bases;
%            LATENT, eigenvalues corresponding to principle components

    % Memorize size of X
    [m, N] = size(X);

    % Shift each sample to 0 mean and Calculate covariance matrix
    standX = bsxfun(@minus, X, mean(X));
    covXT = (standX * standX');
    
    % D is diag eigenvalues matrix in increasing order
    % V is the eigenvectors of covXT in corresponding order
    [V, D] = eig(covXT);
    
    % LATENT is eigenvalues covX in descending order
    % COEFF is Re-unitize eigenvectors in corresponding order
    COEFF = standX' * fliplr(V);
    LATENT = flip(diag(D)) / (m-1);
    
    % Calculate k to retain 95% accuracy
    tempSum = 0;
    eigSum = sum(LATENT);
    k = 0;
    while tempSum <= 0.95 * eigSum
        k = k + 1;
        tempSum = tempSum + LATENT(k);
    end
    LATENT = LATENT(1:k);
    COEFF = COEFF(:,1:k);
    
    % Re-unitize eigenvectors in COEFF
    normCOEFF = zeros(1, k);
    for i = 1:k
        normCOEFF(1,i) = norm(COEFF(:,i));
    end
    COEFF = COEFF ./ repmat(normCOEFF, N, 1);
    
    % Re-base shifted X to the COEFF bases to get SCORE
    SCORE = (standX * COEFF);
    SCORE = SCORE(:,1:k);

end