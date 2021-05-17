
## Code ##


### Pipenv <h3> 
```
pipenv install
```
```
pipenv shell
```
### Install <h3> 
```
pipenv install requests
```
### Run the code <h3> 
```  
pipenv run python main.py
```
  
## Model ##
使用時間序列:GRU去做每天每小時的產電量以及消耗電量的預測,

將所有household做為資料集做訓練,

以七天,共168小時作為input,

未來一天24小時作為預測目標。


## Method ##

讀取gen和con兩個七天前的資料得出未來一天24小時的產電以及消耗電量

相減後得到每個小時的多餘電量(或需求電量)。

採用的策略為若此小時預測為需求電量,假設其為 x ,

則以比台電較為低的價格購買 x 度,

同時我將此小時產出的電量嘗試以比台電高的價格售出以此獲利。

若預測為多餘電量,則以較為低廉的價格賣出 x 度。


