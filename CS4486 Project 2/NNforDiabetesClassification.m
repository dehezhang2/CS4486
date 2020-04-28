%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The code is to be run using MATLAB version 6.5 or other higher versions.
%This program trains the Multi-Layer Perceptrons for the diabetes classification 
%problem using gradient descent based backpropagation. The MLPs architecture used 
%here consist of a single hidden layer and an output layer,and both layers use the 
%sigmoid neurons. There are eight input neurons for the diabetes input feature and 
%one input neuron for the bias. There are two output neurons, and the decided class 
%of the MLPs uses the winner-take-all strategy, i.e., the output class predicted by the 
%network is the corresponding output neuron with the highest value. The number of hidden 
%neurons used by the MLPs is varied from one to a maximum number pre-defined by the user.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The original data file from the UCI repository has been preprocessed to be
%used for this program. All feature values have been normalised. 
%The first column of the diabetes data file is the bias input, and the
%second to the ninth columns are the input features. The tenth and eleventh
%columns of the diabetes data file are the output values. A one in the tenth column 
%would represent diabetes positive. There are a total of 768 patterns in
%the dataset. 576 are used as training pattern and 192 are used as test
%pattern. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function MLP()

close all
clear all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initialisation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
patternNum=768;
trnDataRatio=0.75; %Percentage of the whole dataset used as training dataset
inputNBiasNum=9;  %Number of input units plus one for bias input
outputNum=2;


diabetesData = load('.\diabetes.txt'); %Change the directory to the one on your computer

trnInData=diabetesData(1:ceil(trnDataRatio*patternNum),1:inputNBiasNum); %Training input data
trnOutData=diabetesData(1:ceil(trnDataRatio*patternNum),inputNBiasNum+1:inputNBiasNum+outputNum); %Training output data
tstInData=diabetesData(ceil(trnDataRatio*patternNum+1):patternNum,1:inputNBiasNum); %Test input data
tstOutData=diabetesData(ceil(trnDataRatio*patternNum+1):patternNum,inputNBiasNum+1:inputNBiasNum+outputNum); %Test output data

[trnPatternNum,inputNum]=size(trnInData);
[trnPatternNum,outputNum]=size(trnOutData);
[tstPatternNum,inputNum]=size(tstInData);

maxRun=5; %number of runs
maxHuNum= 10; %maximum number of hidden neurons to be used. 
maxEpoch=60; 
learningRate = 0.9; 

for run=1:maxRun
    run
for huNum=1:1:maxHuNum
    huNum
     
    epoch = 1; 
    w1=2*rand(huNum,inputNum)-1; % weights of inputs and bias to hidden units 
    w2=2*rand(outputNum,huNum+1)-1; %weights of hidden units and bias to output units
    clear huInput huOutput ouInput ouOutput
    startTime=cputime;
    while(epoch<=maxEpoch)
        outSubErr=zeros(1,outputNum); tErr = 0; trnCorr = 0;tstCorr = 0;
        for patternCount=1:trnPatternNum
           %%%%%%%%%forward pass%%%%%%%%%%%%%%
           for i=1:huNum %hidden layer
                huInput(i)=trnInData(patternCount,:)*w1(i,:)';
                huOutput(i)=logsig(huInput(i));
           end

           for i=1:outputNum %output layer
                ouInput(i)= w2(i,:)*[1;huOutput'];
                ouOutput(patternCount,i)= logsig(ouInput(i)); 
            end

           %%%%%%%%%%%%%%backward pass%%%%%%%%%
           for i=1:outputNum
               outputLocalError(i)=(trnOutData(patternCount,i)-ouOutput(patternCount,i))*ouOutput(patternCount,i)*(1-ouOutput(patternCount,i));
           end
           for i=1:huNum
               huLocalError(i)=huOutput(i)*(1-huOutput(i))*outputLocalError(1,:)*w2(:,i+1);
           end

           %%%%%%%%%weights update%%%%%%%%%%%%%%
           for i=1:outputNum
           w2(i,:)=w2(i,:)+learningRate*outputLocalError(i)*[1;huOutput']';
           end
           for i=1:huNum
               w1(i,:)= w1(i,:)+learningRate*huLocalError(i)*trnInData(patternCount,:);
           end

           %Based on sum of squared error
           for i=1:outputNum
            outSubErr(i)=outSubErr(i)+0.5*(trnOutData(patternCount,i)-ouOutput(patternCount,i))^2;
           end
 
        end %of one epoch 
        
        for i=1:outputNum
            tErr=tErr+outSubErr(i); %total error for all output during one epoch pass
        end
        Err(run,epoch,huNum) = tErr;
        
        %%%%%%%%%Calculate classification accuracy on Trn set%%%%%%%%%%%%%
           for patternCount=1:trnPatternNum
               for i=1:huNum %hidden layer
                huInput(i)=trnInData(patternCount,:)*w1(i,:)';
                huOutput(i)=logsig(huInput(i));
               end
               for i=1:outputNum %output layer
                ouInput(i)= w2(i,:)*[1;huOutput'];
                ouOutput(patternCount,i)= logsig(ouInput(i)); 
               end
                winningClassTrn=1;
               for i=2:outputNum
                 if(ouOutput(patternCount,i)>ouOutput(patternCount,1))&(ouOutput(patternCount,i)>ouOutput(patternCount,winningClassTrn))
                 winningClassTrn=i;
                 end
               end
               if trnOutData(patternCount,winningClassTrn)== 1
                  trnCorr=trnCorr+1;
               end
           end
             trnCorrPercent = (trnCorr/trnPatternNum)*100;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        trnAccuracy(run,epoch,huNum)=trnCorrPercent;
     
 
       epoch = epoch+1;
    end  %maxEpoch 
    
    endTime=cputime;
    time(run,huNum)=(endTime-startTime);
    endTime=0;
    startTime=0;
    
    %%%%%%%%%Calculate generalization accuracy on Tst set%%%%%%%%%%
        for patternCount=1:tstPatternNum
         for i=1:huNum %hidden layer
              huInput(i)=tstInData(patternCount,:)*w1(i,:)';
              huOutput(i)=logsig(huInput(i));
         end
         for i=1:outputNum %output layer
              ouInput(i)= w2(i,:)*[1;huOutput'];
              ouOutput(patternCount,i)= logsig(ouInput(i)); 
         end     
         winningClass=1;
         for i=2:outputNum
             if(ouOutput(patternCount,i)>ouOutput(patternCount,1))&(ouOutput(patternCount,i)>ouOutput(patternCount,winningClass))
             winningClass=i;
             end
         end
          if tstOutData(patternCount,winningClass)== 1
              tstCorr=tstCorr+1;
          end
        end
        tstCorrPercent = (tstCorr/tstPatternNum)*100;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       tstAccuracy(run,huNum)=tstCorrPercent;
       
end %maxHuNum
end %maxRun
save Results


%%%%%%%%Plot Sum of Squared Error%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for epoch=1:maxEpoch
    for huNum=1:maxHuNum
        avgErr(epoch,huNum)=mean(Err(:,epoch,huNum));
    end
end
figure (1);
surf(avgErr);
colormap(winter);
xlabel(['Number of hidden units']);
ylabel('Number of epochs');
zlabel('Sum of Squared Error');


%%%%%%%%Plot Training Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for epoch=1:maxEpoch
    for huNum=1:maxHuNum
    avgTrnAcc(epoch,huNum)=mean( trnAccuracy(:,epoch,huNum));
    end
end
figure (2);
surf(avgTrnAcc);
colormap(winter);
xlabel(['Number of hidden units']);
ylabel('Number of epochs');
zlabel('Accuracy on training set (%)');

%%%%%%%%Plot Test Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for huNum=1:maxHuNum
    avgTstAcc(huNum)=mean(tstAccuracy(:,huNum));
    maxTstAcc(huNum)=max(tstAccuracy(:,huNum));
    minTstAcc(huNum)=min(tstAccuracy(:,huNum));
end
figure (3),
plot([1:1:maxHuNum],maxTstAcc,'ro-',[1:1:maxHuNum],avgTstAcc,'b^-',[1:1:maxHuNum],minTstAcc,'g+-'),
xlabel('Number of hidden units');
ylabel('Classification Accuracy (%)');
legend('Max Accuracy','Mean Accuracy','Min Accuracy');


%%%%%%%%Plot Time graph %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for huNum=1:maxHuNum
   avgTime(huNum)=mean(time(:,huNum));
   maxTime(huNum)=max(time(:,huNum));
   minTime(huNum)=min(time(:,huNum));
end
figure (4),
plot([1:1:maxHuNum],maxTime,'ro-',[1:1:maxHuNum],avgTime,'b^-',[1:1:maxHuNum],minTime,'g+-'),
xlabel('Number of hidden units');
ylabel('Time in seconds');
legend('Max Time','Mean Time','Min Time');

end
