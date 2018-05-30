if("sigmoid" %in% rownames(installed.packages()) == FALSE) {
  install.packages("sigmoid")
}
require(sigmoid)

mapVectorForRnn <- function(wieghts) {
  s = 1
  e = inputSize * hiddenSize
  map = list()
  map$hiddenWieghts <- matrix(wieghts[s:e], 
                              ncol = hiddenSize, 
                              byrow = TRUE)
  
  s = e + 1
  e = e + hiddenSize * hiddenSize
  
  map$recurrentWieghts <- matrix(wieghts[s:e], 
                                 ncol = hiddenSize,
                                 byrow = TRUE)
  
  s = e + 1
  e = e + outSize * hiddenSize
  map$outputWieghts <- matrix(wieghts[s:e], 
                              ncol = outSize, 
                              byrow = TRUE)
  return(map)
}

mapVectorForLstm <- function(wieghts) {
  s = 1
  e = inputSize * hiddenSize
  map = list()
  map$Wi <- matrix(wieghts[s:e], 
                              nrow = hiddenSize, 
                              byrow = TRUE)
  
  s = e + 1
  e = e + hiddenSize * hiddenSize
  
  map$Ui <- matrix(wieghts[s:e], 
                   nrow = hiddenSize,
                   byrow = TRUE)

  s = e + 1
  e = e + inputSize * hiddenSize
  map$Wf <- matrix(wieghts[s:e], 
                   nrow = hiddenSize, 
                   byrow = TRUE)
  s = e + 1
  e = e + hiddenSize * hiddenSize
  
  
  map$Uf <- matrix(wieghts[s:e], 
                   nrow = hiddenSize,
                   byrow = TRUE)
  s = e + 1
  e = e + inputSize * hiddenSize
  map$Wo <- matrix(wieghts[s:e], 
                   nrow = hiddenSize, 
                   byrow = TRUE)
  
  
  s = e + 1
  e = e + hiddenSize * hiddenSize
  map$Uo <- matrix(wieghts[s:e], 
                   nrow = hiddenSize,
                   byrow = TRUE)  

  
  s = e + 1
  e = e + inputSize * hiddenSize
  map$Wc <- matrix(wieghts[s:e], 
                   nrow = hiddenSize, 
                   byrow = TRUE)
  
  s = e + 1
  e = e + hiddenSize * hiddenSize
  map$Uc <- matrix(wieghts[s:e], 
                   nrow = hiddenSize,
                   byrow = TRUE)  
  
  s = e + 1
  e = e + outSize * hiddenSize
  map$outputWieghts <- matrix(wieghts[s:e], 
                              nrow = outSize, 
                              byrow = TRUE)
  return(map)
}


rnnClassify <- function(hiddenWieghts, recurrentWieghts, outputWieghts, recurrent){
  result <- c()
  
  for(i in 1:length(dataset[,1])){
    
    sample = as.double(dataset[i,])
    
    inputResult <- sample %*% hiddenWieghts + recurrent %*% recurrentWieghts
    inputResult <- sigmoid(inputResult)
    
    recurrent <- inputResult
    
    output <- inputResult %*% outputWieghts
    output <- ifelse(output < 0.5, 0, 1)
    
    output <-  sum(2^(0:(outSize-1)) * output)
    result <- c(result, output[1])
  }
  return(result)
}

lstmClassify <- function(Wi, Ui, Wf, Uf, Wo, Uo, Wc, Uc, outputWieghts){

  result <- c()
  
  for(index in 1:length(dataset[,1])){
    Ht_1 <- rep(0, hiddenSize)
    ct_1 <- rep(0, hiddenSize)
    
    XX <- matrix(as.double(dataset[index,]), ncol = 1)
    XX <- XX[-(1:2)]
    
    for (tt in seq(1,170, 10)){
      X <- XX[tt:(tt+9)]

      it <-  sigmoid(Wi %*% X + Ui %*% Ht_1)
      ft <-  sigmoid(Wf %*% X + Uf %*% Ht_1)
      ot <-  sigmoid(Wo %*% X + Uo %*% Ht_1)
      ct_ <- tanh(Wc %*% X + Uc %*% Ht_1)
    
      ct <- ft * ct_1 + it * ct_
      ht <- ot * tanh(ct)
    
      ct_1 <- ct
      Ht_1 <- ht
    }
    
    output <- outputWieghts %*% ht
    output <- ifelse(output < 0.5, 0, 1)
    
    output <-  sum(2^(0:(outSize-1)) * output)
    result <- c(result, output[1])
  }
  return(result)
}

classify <- function(wieghts, nural = "lstm"){
  if(nural == "lstm"){
    map = mapVectorForLstm(wieghts)
    result <- lstmClassify(map$Wi, map$Ui, map$Wf, map$Uf, map$Wo, map$Uo, map$Wc, map$Uc, map$outputWieghts)
  }
  else{
    map = mapVectorForRnn(wieghts)
    result <- rnnClassify(map$hiddenWieghts, map$recurrentWieghts, map$outputWieghts, rep(0, hiddenSize))
  }
  
  return(result)
}