function [s, R] = test_rfwr_nD(s)global sdcs;% test for RFWR on n-dimensional data set with 2D embedded functiond = 2;n = 500;% a random training set using the CROSS functionX = (rand(n,2)-.5)*2;Y = max([exp(-X(:,1).^2 * 10),exp(-X(:,2).^2 * 50),1.25*exp(-(X(:,1).^2+X(:,2).^2)*5)]');Y = Y' + randn(n,1)*0.1;% rotate input data into d-dimensional spaceif ~exist('R') || isempty(R)  R = randn(d);  R = R'*R;  R = orth(R);endXorg = X;X = [X zeros(n,d-2)]*R;##X = dlmread('trainX.csv', ',', 0, 0)##Y = dlmread('trainY.csv', ',', 0, 0)% a systematic test set on a gridXt = [];for i=-1:0.05:1,	for j=-1:0.05:1,		Xt = [Xt; i j];	endendYt = max([exp(-Xt(:,1).^2 * 10),exp(-Xt(:,2).^2 * 50),1.25*exp(-(Xt(:,1).^2+Xt(:,2).^2)*5)]');Yt = Yt';% rotate the test dataXtorg = Xt;Xt = [Xt zeros(length(Xt),d-2)]*R;% initialize RFWR using only diagonal distance metricID = 1;if ~exist('s') || isempty(s)  rfwr('Init',ID,d,1,1,0,0,1e-6,50,ones(d,1),[1],'rfwr_test');else  rfwr('Init',ID,s);end% set some parameterskernel = 'Gaussian';% kernel = 'BiSquare'; % note: the BiSquare kernel requires different values for%                              an initial distance metric, as in the next line% rfwr('Change',ID,'init_D',eye(d)*7);rfwr('Change',ID,'init_D',eye(d)*25); rfwr('Change',ID,'init_alpha',ones(d)*250);     % this is a safe learning raterfwr('Change',ID,'w_gen',0.2);                  % more overlap gives smoother surfacesrfwr('Change',ID,'meta',1);                     % meta learning can be faster, but numerical more dangerousrfwr('Change',ID,'meta_rate',250);% train the modelfor j=1:20	inds = randperm(n);	mse = 0;	for i=1:n,		[yp,w] = rfwr('Update',ID,X(inds(i),:)',Y(inds(i),:)');		mse = mse + (Y(inds(i),:)-yp).^2;	end	nMSE = mse/n/var(Y,1);		disp(sprintf('#Data=%d #rfs=%d nMSE=%5.3f',sdcs(ID).n_data,length(sdcs(ID).rfs),nMSE));end% create predictions for the test dataYp = zeros(size(Yt));for i=1:length(Xt),	[yp,w]=rfwr('Predict',ID,Xt(i,:)',0.001);	Yp(i,1) = yp;endep   = Yt-Yp;mse  = var(ep,1);nmse = mse/var(Y,1);% get the data structuresdc = rfwr('Structure',ID);figure(1);clf;% plot the raw noisy datasubplot(2,2,1);plot3(Xorg(:,1),Xorg(:,2),Y,'*');title('Noisy data samples');% plot the fitted surfaceaxis([-1 1 -1 1 -.5 1.5]);subplot(2,2,2);[x,y,z]=makesurf([Xtorg,Yp],sqrt(length(Xtorg)));surfl(x,y,z);axis([-1 1 -1 1 -.5 1.5]);title(sprintf('The fitted function: nMSE=%5.3f',nmse));% plot the true surfacesubplot(2,2,3);[x,y,z]=makesurf([Xtorg,Yt],sqrt(length(Xtorg)));surfl(x,y,z);axis([-1 1 -1 1 -.5 1.5]);title('The true function');% plot the local modelssubplot(2,2,4);for i=1:length(sdcs(ID).rfs),  D = R'*sdcs(ID).rfs(i).D*R;  c = R*sdcs(ID).rfs(i).c;	draw_ellipse(D(1:2,1:2),c(1:2),0.1,kernel);	hold on;endhold off;axis('equal');title('Projected space view of RFs');% --------------------------------------------------------------------------------function [X,Y,Z]=makesurf(data,nx)% [X,Y,Z]=makesurf(data,nx) converts the 3D data file data into% three matices as need by surf(). nx tells how long the row of the% output matrices are[m,n]=size(data);n=0;for i=1:nx:m,	n = n+1;	X(:,n) = data(i:i+nx-1,1);	Y(:,n) = data(i:i+nx-1,2);	Z(:,n) = data(i:i+nx-1,3);end;% --------------------------------------------------------------------------------function []=draw_ellipse(M,C,w,kernel)% function draw ellipse draws the ellipse corresponding to the% eigenvalues of M at the location c.[V,E] = eig(M);E = E;d1 = E(1,1);d2 = E(2,2);steps = 50;switch kernelcase 'Gaussian'	start = sqrt(-2*log(w)/d1);case 'BiSquare'	start = sqrt(2*(1-sqrt(w))/d1);endfor i=0:steps,	Xp(i+1,1) = -start + i*(2*start)/steps;	switch kernel	case 'Gaussian'		arg = (-2*log(w)-Xp(i+1,1)^2*d1)/d2;	case 'BiSquare'		arg = (2*(1-sqrt(w))-Xp(i+1,1)^2*d1)/d2;	end	if (arg < 0), 		arg = 0; 	end; % should be numerical error	Yp(i+1,1) = sqrt(arg);end;for i=1:steps+1;	Xp(steps+1+i,1) = Xp(steps+1-i+1,1);	Yp(steps+1+i,1) = -Yp(steps+1-i+1,1);end;% transform the rfM = [Xp,Yp]*V(1:2,1:2)';Xp = M(:,1) + C(1);Yp = M(:,2) + C(2);plot(C(1),C(2),'ro',Xp,Yp,'c');