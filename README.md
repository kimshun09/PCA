# Principal Component Analysis

C++で主成分分析(PCA)をやってみた．

`asg03/main.cpp`では，`asg03/iris.dat`のデータを用いて主成分分析を行い，第1主成分，第2主成分のスコアを求める．

`bool eig(const MatrixXd& M, VectorXd& eval, MatrixXd& evec)`によって，行列`M`を受け取って，2つ目の引数に昇順に並んだ固有値を，3つ目の引数に固有値に対応する順番で各列に固有ベクトルを含む行列を作成する．

また，`bool plot(const MatrixXd& data, const VectorXd& Y)`によって$n\times2$の行列変数`data`と3種類のラベルを含むベクトル`Y`を受け取ってラベルごとに色付けしたプロットを保存する．

このプログラムによる実行結果(第1，第2主成分ベクトルとその寄与率)は以下のようになる．

```NO
1st principal component vector : 0.522372 -0.263355 0.581254 0.565611
2nd principal component vector : -0.372318 -0.925556 -0.0210948 -0.0654158 contribution ratio of 1st principal component vector :0.727705
contribution ratio of 2nd principal component vector :0.230305
```

また，作成されるプロットは以下のようになる．

![graph](graph/plot.eps)