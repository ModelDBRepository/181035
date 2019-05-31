function O = FFV1MT(II,n_scales,th,th2,vel,n_filters,D)

%%% v0.1
%%% 26/03/2015
%%%
%%% M.Chessa and F. Solari
%%% University of Genoa, ITALY
%%%
%%% manuela.chessa@unige.it
%%% fabio.solari@unige.it
%%%
%%% REF PAPER:
%%% F. Solari, M. Chessa, N. Medathati, and P. Kornprobst. 
%%% What can we expect from a V1-MT feedforward architecture for optical flow estimation? 
%%% Submitted to Signal Processing: Image Communication, 2015.
%%%
%%% INPUTS:
%%% II: image sequence [m X n X 5]
%%% n_scales: number of spatial scales
%%% th: motion energy threshold
%%% th2: motion energy threshold after spatial pooling
%%% vel: component velocities V1 level
%%% n_filters: spatial orientation V1 level
%%% D: speed directions MT level
%%%
%%% OUTPUTS:
%%% O: computed optic flow [m X n X 2]


n_frames = size(II,3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change Image Size for Pyramid Construction %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[sy1 sx1 st] = size(II);

fac = 2^(n_scales-1);

sy2 = ceil(sy1 ./ fac) .* fac; % target resolution
sx2 = ceil(sx1 ./ fac) .* fac; % target resolution

II = [ II ; repmat(II(end,:,:),[sy2-sy1 1 1]) ]; % replicate border row
II = [ II repmat(II(:,end,:),[1 sx2-sx1 1]) ]; % replicate border column


%%%%%%%%%%%%%%%%%
% Image Pyramid %
%%%%%%%%%%%%%%%%%

[II] = image_pyramid(II,n_frames,n_scales);

%%%%%%%%%%%%%%%%%%%%%%%%%
% Level 1 full velocity %
%%%%%%%%%%%%%%%%%%%%%%%%%

F = filt_gabor_space(II{1},n_filters);    
F = filt_gabor_time(F,vel);

Os= V1_MT(F,II{1},th,th2,vel,D);
O=Os;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Coarse-to-fine Estimation and Merging %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for scale = 2:n_scales
    
    O = expand(O.*2);
        
    F = filt_gabor_space(II{scale},n_filters);
    
    V = distribute_optic(O,n_frames);
    F = warp_sequence(F,V);
    F{1,1}(isnan(F{1,1}))=0;
    F{1,2}(isnan(F{1,2}))=0;
    F = filt_gabor_time(F,vel);

    Os= V1_MT(F,II{scale},th,th2,vel,D);
    clear F;
    
    O = merge_flow(O,Os);
    
end



% Remove all flow that has not been updated (confirmed) on the lowest
% scale

IND = isnan(sum(Os,3));
O(cat(3,IND,IND)) = NaN;

% Remove rows and columns that were added to construct the pyramid

O(end-(sy2-sy1-1):end,:,:) = [];
O(:,end-(sx2-sx1-1):end,:) = [];





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [II,x_pix,y_pix] = image_pyramid(II,n_frames,n_scales)


[sy, sx, st] = size(II);

x_pix = cell(1,n_scales);
y_pix = cell(1,n_scales);

[ x_pix{n_scales}, y_pix{n_scales} ] = meshgrid(1:sx,1:sy);

lpf = [1 4 6 4 1]/16;

tmp = II;
II = cell(1,n_scales);
II{n_scales} = tmp;

for scale = n_scales-1:-1:1
    for frame = 1:n_frames
        tmp(:,:,frame) = conv2b(conv2b(tmp(:,:,frame),lpf),lpf');
    end
    [Ny, Nx, dummy] = size(tmp);

    tmp = tmp(1:2:Ny,1:2:Nx,:);
    II{scale} = tmp;
    x_pix{scale} = x_pix{scale+1}(1:2:Ny,1:2:Nx);
    y_pix{scale} = y_pix{scale+1}(1:2:Ny,1:2:Nx);
end



%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%


function O = expand(O)


sy = size(O,1);
sx = size(O,2);

[X Y] = meshgrid(1:(sx-1)/(2*sx-1):sx, ...
    1:(sy-1)/(2*sy-1):sy);


% Repeat edge pixel for border handling

O = [ O O(:,end,:) ];
O = [ O ; O(end,:,:) ];

O = bilin_interp(O,X,Y);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function G_warped = warp_sequence(G,V)


[sy, sx, n_orient, n_frames] = size(G{1});
G_warped = cell(1,2);

[X, Y] = meshgrid(1:sx,1:sy);
tmp = V;
tmp(isnan(tmp)) = 0;
Vx = tmp(:,:,:,1);
Vy = tmp(:,:,:,2) ;

for frame = 1:n_frames

    Xn = X-Vx(:,:,frame);
    Yn = Y-Vy(:,:,frame);
    G_warped{1}(:,:,:,frame) = (bilin_interp((squeeze(G{1}(:,:,:,frame))),Xn,Yn));
    G_warped{2}(:,:,:,frame) = (bilin_interp((squeeze(G{2}(:,:,:,frame))),Xn,Yn));

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function O = merge_flow(O1,O2)


invalid1 = isnan(sum(O1,3));
invalid2 = isnan(sum(O2,3));

O1(cat(3,invalid1,invalid1)) = 0;
O2(cat(3,invalid2,invalid2)) = 0;

invalid = invalid1 & invalid2;

textured=0;
if ~textured
    invalid = invalid1 | invalid2;
end

O = O1 + O2;
O(cat(3,invalid,invalid)) = NaN;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function V = distribute_optic(O,n_frames)

[sy, sx, dummy]=size(O);
V = zeros(sy,sx,n_frames,2);
O(isnan(O)) = 0;

sh=[2 1 0 -1 -2];
for ii=1:n_frames
    V(:,:,ii,1) =O(:,:,1)*sh(ii);   V(:,:,ii,2) = O(:,:,2)*sh(ii);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [O]= V1_MT(F,II,th,th2,vel,D)

n_vel=size(F,2);


[sy, sx, n_orient]=size(F{1,1});
E=zeros(sy,sx,n_orient,n_vel);
E_v1=zeros(sy,sx,n_orient,n_vel);
if D==2
    E_mt=zeros(sy,sx,D,n_vel);
else
    E_mt=zeros(sy,sx,D-2,n_vel);
end


for n_v=1:n_vel
    E(:,:,:,n_v)= sqrt(F{1,n_v}.^2+F{2,n_v}.^2);
end

E=E.^0.5;

E=E/max(max(max(max(E))));
mask=E>th;
E=E.*mask;

O = NaN(sy, sx, 2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%normalization: tmp matrix used later
tmp=zeros(sy,sx,n_vel);
for v=1:n_vel
    for o=1:n_orient
        tmp(:,:,v)=tmp(:,:,v)+E(:,:,o,v);
    end
end


filt = fspecial('gaussian', [5 5], 10/6);
filt=filt./sum(sum(fspecial('gaussian', [5 5], 10/6)));


for v=1:n_vel
    for o=1:n_orient
        E_v1(:,:,o,v)=E(:,:,o,v)./(tmp(:,:,v)+1e-9);%normalization
        E_v1(:,:,o,v) = conv2(E_v1(:,:,o,v), filt, 'same');%V1 spatial pooling
        mask=ones(sy,sx);
        mask(tmp(:,:,v)<th2)=NaN;%threshold (unreliable pixels)
        E_v1(:,:,o,v)=E_v1(:,:,o,v).*mask;
    end
end

E_v1=0.25*E_v1/max(max(max(max(E_v1))));%exponential gain

%%%%%%orientation pooling
if D==2
    vdir=[0,pi/2];
else
    vdir=-pi:(2*pi)/(D-1):pi;
    vdir(1)=[];
    vdir(end)=[];
    D=D-2;
end

for ivd=1:D
    for v=1:n_vel
        csum=zeros(sy,sx);
        for o=1:n_orient
            theta=(o-1)*(2*pi/n_orient);
            csum=csum + cos(vdir(ivd)-theta)*E_v1(:,:,o,v);
        end
        E_mt(:,:,ivd,v)=exp(csum);
    end
end

VD=zeros(sy,sx,D);
vx=zeros(sy,sx);
vy=zeros(sy,sx);


%%%%%%%%%interpolation: filling-in of borders and unreliable pixels
textured=1;
if textured
    MMi=max(max(max(II(:,:,3))));
    mmi=min(min(min(II(:,:,3))));
    
    [xx1,xx2,xx3]=size(E_mt(:,:,1));
    
    if xx1<20 && (~isequal(isnan(E_mt),zeros(size(E_mt))) )
        for ivd=1:D
            for v=1:n_vel
                Os=E_mt(:,:,ivd,v);
                Os=fillin_ppp(Os,II(:,:,3),(MMi-mmi)/6);
                E_mt(:,:,ivd,v)=Os;
            end
        end
    end
    
    if xx1>=20
        for ivd=1:D
            for v=1:n_vel
                Os=E_mt(:,:,ivd,v);
                Os(1:5,:,:)=NaN; Os((end-4):end,:,:)=NaN;  Os(:,1:5,:)=NaN; Os(:,(end-4):end,:)=NaN;
                Os=fillin_ppp(Os,II(:,:,3),(MMi-mmi)/6);
                E_mt(:,:,ivd,v)=Os;
            end
        end
    end
end
%%%%%%%%%%%%%%%

%%%%%%MT decoding
%%%%component
for ivd=1:D
    for v=1:n_vel
        VD(:,:,ivd)=VD(:,:,ivd)+E_mt(:,:,ivd,v)*vel(v);
    end
end

%%%%argmin
for ivd=1:D
    vx=vx + VD(:,:,ivd)*cos(vdir(ivd));
    vy=vy + VD(:,:,ivd)*sin(vdir(ivd));
end
vx=(2/D)*vx;
vy=(2/D)*vy;

O(:,:,1)=vx;
O(:,:,2)=vy;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function IF = filt_gabor_space(I,n_filters)

[sy,sx,n_frames] = size(I);

IF{1} = zeros(sy,sx,n_filters,n_frames);
IF{2} = zeros(sy,sx,n_filters,n_frames);


w=5;
f0=1/3.8;
sigma_s=1.2*sqrt(2*log(2)) ./ (2*pi*(f0/3));


[X,Y] = meshgrid(-w:w,-w:w);
theta=0:2*pi/n_filters:(2*pi-2*pi/n_filters);

G = exp(-(X.^2+Y.^2)/(2*sigma_s^2));

for ii=1:length(theta)
    XT=cos(theta(ii))*X+sin(theta(ii))*Y;
    GC=G.*cos(2*pi*f0*XT);
    GCB{ii}=GC-sum(sum(GC))/(2*w+1)^2;%DC
    GS=G.*sin(2*pi*f0*XT);
    GSB{ii}=GS-sum(sum(GS))/(2*w+1)^2;%DC
end


for frame = 1:n_frames

    for ii=1:n_filters/2        
        even=conv2b(I(:,:,frame),GCB{ii});
        odd=conv2b(I(:,:,frame),GSB{ii});
        
        IF{1}(:,:,ii,frame) = even;
        IF{1}(:,:,ii+n_filters/2,frame) = even;
        
        IF{2}(:,:,ii,frame) = odd;
        IF{2}(:,:,ii+n_filters/2,frame) = -odd;
    end
 
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function IF = filt_gabor_time(F,v)


n_vel=size(v,2);
[sy, sx, n_orient, n_frames]=size(F{1});

for n_v=1:n_vel
    
    w0=-v(n_v)/3.8;
    
    
    %%%EXP
    t=0:n_frames-1;
    f0=1/3.8;
    B=f0/2.5;
    sigma = sqrt(2*log(2)) ./ (2*pi*B);
    g = exp(-t ./ (2.*sigma.^2));
    Fts = g.*sin(2*pi*w0*t);
    Ftc = g.*cos(2*pi*w0*t);
    Ftc=Ftc';
    Fts=Fts';    
    
    G_even_tmp=F{1};
    G_odd_tmp=F{2};
    
    G_even3d=zeros(sy,sx,n_orient);
    G_odd3d=zeros(sy,sx,n_orient);
    
    for orient=1:n_orient
        for i=1:sy
            
            G_even3d(i,:,orient) = (conv2(squeeze(G_even_tmp(i,:,orient,:))',Ftc,'valid')-conv2(squeeze(G_odd_tmp(i,:,orient,:))',Fts,'valid'))';
            G_odd3d(i,:,orient) = (conv2(squeeze(G_even_tmp(i,:,orient,:))',Fts,'valid')+conv2(squeeze(G_odd_tmp(i,:,orient,:))',Ftc,'valid'))';
            
        end
    end
    
    IF{1,n_v}= squeeze(G_even3d);
    IF{2,n_v} = squeeze(G_odd3d);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function imf=conv2b(im, ker)
[nky,nkx]=size(ker);
sh='valid';
Bx=(nkx-1)/2;
By=(nky-1)/2;
im=putborde(im,Bx,By);
imf=conv2(im,ker,sh);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function imb=putborde(im,Nx,Ny)

[sy,sx]=size(im);
imb=zeros(sy+2*Ny,sx+2*Nx);
imb(1+Ny:sy+Ny,1+Nx:sx+Nx)=im;

for k=1:Nx
	imb(Ny+1:sy+Ny,k)=im(:,1);
	imb(Ny+1:sy+Ny,k+sx+Nx)=im(:,sx);
end
for k=1:Ny
	imb(k,Nx+1:sx+Nx)=im(1,:);
	imb(k+sy+Ny,Nx+1:sx+Nx)=im(sy,:);
end



function I2 = bilin_interp(I1,X,Y)
% function I2 = bilin_interp(I1,X,Y)
%
% Arbitrary image rewarping using bilinear interpolation
%
% X (column) and Y (row) (both floating point) are the source locations,
% used to fill the respective pixels
  
[nY1,nX1,rem] = size(I1); % source size
[nY2,nX2] = size(X); % target size

s = size(I1);
s(1:2) = [];

I2 = NaN.*zeros([ nY2 nX2 s ]);

for r = 1:rem
    
    for x = 1:nX2
        for y = 1:nY2
            
            % Pixel warping (2x2 group)
            
            x_w = floor(X(y,x));
            y_w = floor(Y(y,x));
            
            % Check validity
            
            if ( (x_w>0) && (x_w<nX1) && (y_w>0) && (y_w<nY1) )
                
                xs = X(y,x) - x_w;
                min_xs = 1-xs;
                ys = Y(y,x) - y_w;
                min_ys = 1-ys;
                
                w_00 = min_xs*min_ys;  % w_xy
                w_10 = xs*min_ys;
                w_01 = min_xs*ys;
                w_11 = xs*ys;
                
                I2(y,x,r) = w_00*I1(y_w,x_w,r) + w_10*I1(y_w,x_w+1,r) + ...
                    w_01*I1(y_w+1,x_w,r) + w_11*I1(y_w+1,x_w+1,r);
                
            end
        end
    end
    
end

function OO=fillin_ppp(Oin,II,sigma_r)
%Boundary conditions&Unreliable regions


w=7;
sigma_d=(2*w+1)/6;
% Pre-compute Gaussian distance weights.
[X,Y] = meshgrid(-w:w,-w:w);
G = exp(-(X.^2+Y.^2)/(2*sigma_d^2));

tmp=Oin; tmp2=Oin;
tmp2(isnan(tmp2))=0;
masknonan=~isnan(tmp);
trueborder = bwmorph(masknonan,'remove');
tmp((trueborder)==0)=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply bilateral filter.
dim = size(II);
tmpi=II;
%tmpi((trueborder)==0)=0;
B = zeros(dim);
for i = 1:dim(1)
   for j = 1:dim(2)
      
         % Extract local region.
         iMin = max(i-w,1);
         iMax = min(i+w,dim(1));
         jMin = max(j-w,1);
         jMax = min(j+w,dim(2));
         I = tmpi(iMin:iMax,jMin:jMax);
         O = tmp(iMin:iMax,jMin:jMax);
         
         % Compute Gaussian intensity weights.
         H = exp(-((I-tmpi(i,j)).^2)/(2*sigma_r^2));
      
         % Calculate bilateral filter response.
         GG=G((iMin:iMax)-i+w+1,(jMin:jMax)-j+w+1);
         F = H.*GG;
         
         N=F.*trueborder(iMin:iMax,jMin:jMax);
         if sum(N(:))==0
             B(i,j) = sum(F(:).*O(:));
         else
            B(i,j) = sum(F(:).*O(:))/sum(N(:));
         end
               
   end
end

O1filled=(B).*(~masknonan) + tmp2.*masknonan;

OO=O1filled;




