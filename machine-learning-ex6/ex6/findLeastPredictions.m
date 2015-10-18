listC = [.01, .03, .1, .03, 1 ,2, 10, 30];
listSigma = [.01, .03, .1, .03, 1 ,2, 10, 30];

% listC = [.01, .03];
% listSigma = [.01, .03];

listError = zeros(length(listC) * length(listSigma), 3 );
count = 1;

for con = listC
	for sig = listSigma
		model= svmTrain(X, y, con, @(x1, x2) gaussianKernel(x1, x2, sig));
		predictions = svmPredict(model, Xval);
		error = mean(double(predictions ~= yval));
		listError(count, :) = [con, sig,  error];
		count = count + 1
	endfor
endfor

indexes = find (listError(:,3) == min(listError)(1,3));
listError(indexes, :)
C = listError(indexes, :)(1,1);
sigma = listError(indexes, :)(1,2);