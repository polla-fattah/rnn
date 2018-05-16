require(pROC)
source('globalSearch.R')

AUC <- function(class1, class2){
  result = tryCatch({
    
  }, warning = function(w) {
    0
  }, error = function(e) {
    0
  })
  return (result)
}


initializeVars <- function(dataFile, lableCol = 1, hSize = 10){
  rawData <<- read.csv(dataFile)
  
  globalDataset <<- rawData[-lableCol]
  globalLabel <<- rawData[lableCol]
  globalLabel <<- as.numeric(t(globalLabel))
  
  inputSize <<- length(globalDataset)
  outSize <<- ceiling(log2(length(table(globalLabel))))
  
  hiddenSize <<- hSize
}

folds <- function (optimizer = harmonySearch, nural="lstm"){
  wieghtsSize = 0
  if(nural == "lstm"){
    wieghtsSize = inputSize * hiddenSize * 4 + hiddenSize * hiddenSize * 4 + hiddenSize * outSize
  }
  else{
    wieghtsSize = inputSize * hiddenSize + hiddenSize * hiddenSize + hiddenSize * outSize
  }
    
  
  testNo = floor(length(globalDataset[,1]) /10)
  testSample = 1:testNo
  results <<- c()
  for (i in 0:9) {
    dataset <<- globalDataset[-(testSample + testNo * i),]
    label <<- globalLabel[-(testSample + testNo * i)]
    
    optWieghts = optimizer(wieghtsSize, nural)
    
    dataset <<- globalDataset[(testSample + testNo * i),]
    label <<- globalLabel[(testSample + testNo * i)]
    pridict <<- classify(optWieghts, nural)
    results[1 + i] <<- sum(label== pridict)/length(label)
    
  }
  return(sum(results) / 10)
}


main <- function(){
  foldResults <<- c()
  file <- "wdbc.csv" #file.choose()
  initializeVars(file, lableCol = 1)
  for(i in 1:1){
    iterations <<- 300 * i
    foldResults[i] <<- folds()
  }
  plot(1:1, foldResults)
  
  
  #print(wieghtsSize)
  #defEvo(wieghtsSize)
  #harmonySearch(wieghtsSize)
  #prticleSwarmOpt(wieghtsSize)
  #antLoinOpt(wieghtsSize)
  
}

main()

