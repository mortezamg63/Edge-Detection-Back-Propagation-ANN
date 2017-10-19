function [re,V,W,V0,W0]=Edge_Detection_BP_ANN(addressImage)
Pattern=16;
Epoch=500;
Nx=4;
Pz=12;
My=4;
Alfa=0.15;
s=[1 1 1 1;1 1 1 0;1 1 0 1;1 1 0 0;1 0 1 1;1 0 1 0;1 0 0 1;1 0 0 0;0 1 1 1;0 1 1 0;0 1 0 1;0 1 0 0;0 0 1 1;0 0 1 0;0 0 0 1;0 0 0 0];
t=[1 1 1 1;1 1 1 1;1 1 1 1;1 1 0 0;1 1 1 1;1 0 1 0;1 0 0 1;1 0 0 1;1 1 1 1;0 1 1 0;0 1 0 1;0 1 1 0;0 0 1 1;0 1 1 0;1 0 0 1;1 1 1 1];

[x,y]=size(t);
h=zeros(x,y);

V=rand(Nx,Pz)-.5;
W=rand(Pz,My)-.5;
V0=rand();
W0=rand();

for E=1:Epoch       
    for P=1:Pattern        
            X=s(P,:);        
        for j=1:Pz
            Sigma=0;
            for i=1:Nx
                Sigma=Sigma+X(i)*V(i,j);
            end
            Zin(j)=V0+Sigma;
            Z(j)=SigmoidBp(Zin(j));
        end
        
        for k=1:My
            Sigma=0;
            for j=1:Pz
                Sigma=Sigma+Z(j)*W(j,k);
            end
            Yin(k)=W0+Sigma;
            Y(k)=(Yin(k));
        end
        
        for k=1:My
            delta_k(k)=(t(P,k)-Y(k)')*PBipSigPrim(Yin(k),1);
            for j=1:Pz
                Delta_W(j,k)=Alfa*delta_k(k)*Z(j);
            end
            Delta_W0(k)=Alfa*delta_k(k);
            W0=W0+Delta_W0(k);
        end
        
        for j=1:Pz
            Sigma=0;
            for k=1:My
                Sigma=Sigma+delta_k(k)*W(j,k);
            end
            delta_in_j(j)=Sigma;
            delta_j(j)=delta_in_j(j)*PBipSigPrim(Zin(j),1);

            for i=1:Nx
                Delta_V(i,j)=Alfa*delta_j(j)*X(i);
            end
            Delta_V0(j)=Alfa*delta_j(j);
            V0=V0+Delta_V0(j);
        end
        
        for k=1:My
            for j=1:Pz
                W(j,k)=W(j,k)+Delta_W(j,k);
            end
        end
        for j=1:Pz
            for i=1:Nx
                V(i,j)=V(i,j)+Delta_V(i,j);
            end
        end
        
    end        
end
mainimage=imread(addressImage);
bwimage=im2bw(mainimage);
[rows,cols]=size(bwimage);
image=ones(rows,cols);
changeimage=zeros(rows,cols);
for i=1:1:rows-1
    for j=1:1:cols-1        
        mask=bwimage(i:i+1,j:j+1);
        inp_NN=[mask(1,1) mask(1,2) mask(2,1) mask(2,2)];
        Z=SigmoidBp(inp_NN*V+V0);
        Y=round(Z*W+W0);
        
        if(image(i,j) ~=0)
            image(i,j)=Y(1);  
            changeimage(i,j)=Y(1);
        end
        if(image(i,j+1)~=0)
            image(i,j+1)=Y(2);
            changeimage(i,j+1)=Y(2);
        end
        if(image(i+1,j)~=0)
            image(i+1,j)=Y(3); 
            changeimage(i+1,j)=Y(3); 
        end
        if(image(i+1,j+1)~=0)
        image(i+1,j+1)=Y(4);
        changeimage(i+1,j+1)=Y(4);
        end
    end    
end
subplot(1,2,1);
imshow(mainimage);
subplot(2,2,2);
imshow(bwimage);
figure;imshow(image);

end


function out = PBipSigPrim(input,mode)

if mode == 1
 
   out=(.5)*((1+PBipSig(input)).*(1-PBipSig(input)));

else

    out=(.5)*((1+input).*(1-input));

end
end

function out = SigmoidBp(input)
out = 2./(1+exp(- input ))-1;
end

function out = PBipSig(input)
out = 2./(1+exp(- input ))-1;
end
