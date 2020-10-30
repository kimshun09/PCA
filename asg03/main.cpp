#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include "Eigen/Core"
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

void scale(MatrixXd& X, VectorXd& mean, VectorXd& std);
bool loadMat(string filename, MatrixXd& m);
MatrixXd cov(const MatrixXd& X);
bool eig(const MatrixXd& M, VectorXd& eval, MatrixXd& evec);
bool plot(const MatrixXd &data, const VectorXd &Y);

int main() {

    MatrixXd Data;
    MatrixXd X;

    loadMat("./iris.dat", Data); // ファイルを変数Dataに読み込む
    X = Data.block(0, 0, Data.rows(), Data.cols()-1); // 最後の列（品種）以外をXへ
    VectorXd Y = Data.col(Data.cols()-1);

    VectorXd mean;
    VectorXd std;

    scale(X, mean, std); // 正規化

    MatrixXd C = cov(X);

    //
    // 固有値分解関数eigを使って主成分ベクトルを求め，寄与率を表示するコードを作成
    //
    // 固有値分解
    VectorXd eval; // 固有値
    MatrixXd evac; // 固有ベクトル
    eig(C, eval, evac);

    // 固有ベクトルの表示
    cout << "1st principal component vector :" << evac.col(0).transpose() << endl;
    cout << "2nd principal component vector :" << evac.col(1).transpose() << endl;

    // 寄与率の計算
    // 固有値の合計を計算し，固有値をそれで割ったものが寄与率である
    double lambda = 0;
    for(int i=0; i<eval.size(); i++)
        lambda += eval(i);
    VectorXd ctbr = eval / lambda;
    // 寄与率の表示
    cout << "contribution ratio of 1st principal component vector :" << ctbr(0) << endl;
    cout << "contribution ratio of 2nd principal component vector :" << ctbr(1) << endl;


    // サイズ n x 2 の行列Uを用意し，U = X*[w1 w2]と設定してプロット関数に渡す
    // MatrixXd U;
    // plot(U, Y);

    // W = [w1 w2]として初期化
    MatrixXd W = MatrixXd::Zero(evac.rows(), 2);
    W.col(0) = evac.col(0);
    W.col(1) = evac.col(1);

    // n x 2行列UをU=XWとしてplot()に渡す
    MatrixXd U = X * W;
    plot(U, Y);

    return 0;

}

MatrixXd cov(const MatrixXd& X) {

    VectorXd mean = VectorXd::Zero(X.cols());
    MatrixXd C = MatrixXd::Zero(X.cols(),X.cols());

    for(int i = 0; i < X.rows(); ++i) {
        mean = mean + X.row(i).transpose();
    }
    mean = mean / X.rows();

    for(int i = 0; i < X.rows(); ++i) {
        C += (X.row(i).transpose() - mean) * (X.row(i) - mean.transpose());
    }

    C = C / X.rows();

    return C;

}

void scale(MatrixXd& X, VectorXd& mean, VectorXd& std) {

    mean = VectorXd::Zero(X.cols());
    std = VectorXd::Zero(X.cols());

    for(int i = 0; i < X.rows(); ++i) {
        mean = mean + X.row(i).transpose();
    }
    mean = mean / X.rows();
    for(int i = 0; i < X.rows(); ++i) {
        X.row(i) = X.row(i) - mean.transpose();
        std = std + VectorXd(X.row(i).array().pow(2.0));
    }
    std = (std / X.rows()).array().sqrt();
    for(int i = 0; i < X.cols(); ++i) {
        X.col(i) = X.col(i) / std(i);
    }

}

bool loadMat(string filename, MatrixXd& m) {
    ifstream input(filename.c_str());
    if (input.fail()) {
        cerr << "ERROR. Cannot find file '" << filename << "'." << endl;
        m = MatrixXd::Zero(1,1);
        return false;
    }

    string line;
    double d;
    vector<double> v;
    int n_rows = 0;
    while (getline(input, line)) {
        ++n_rows;
        stringstream input_line(line);
        while (!input_line.eof()) {
            input_line >> d;
            v.push_back(d);
        }
    }
    input.close();

    int n_cols = v.size()/n_rows;
    m = MatrixXd(n_rows,n_cols);
    for (int i=0; i<n_rows; i++)
        for (int j=0; j<n_cols; j++)
            m(i,j) = v[i*n_cols + j];

    return true;
}

struct data {
    double value;
    int index;
};

struct by_value {
    bool operator()(data const &left, data const &right) {
        return left.value > right.value;
    }
};

bool eig(const MatrixXd& M, VectorXd& eval, MatrixXd& evec) {
    SelfAdjointEigenSolver<MatrixXd> es(M);
    eval = es.eigenvalues();
    evec = es.eigenvectors();

    // vector<double> val(eval.data(), eval.data() + eval.size());
    // sort(val.begin(), val.end(), greater<double>());

    vector<data> items(eval.size());
    for(int i = 0; i < eval.size(); ++i) {
        items[i].value = eval(i);
        items[i].index = i;
    }

    MatrixXd evec_temp = evec;
    sort(items.begin(), items.end(), by_value());
    for(int i = 0; i < eval.size(); ++i) {
        eval(i) = items[i].value;
        evec.col(i) = evec_temp.col(items[i].index);
    }

    if (es.info() != Success) return false;

    return true;
}

//bool plot(const MatrixXd &data, const VectorXd &Y) {
//
//    if(data.cols() != 2) {
//        cerr << "data should be two-dimension" << endl;
//        return false;
//    }
//
//    FILE *fp = popen("gnuplot", "w");
//    if (fp == NULL)
//        return false;
//
//    ofstream ofs;
//    fprintf(fp,"set terminal postscript eps enhanced color\n");
//    fprintf(fp,"set output 'plot.eps'\n");
//    fprintf(fp,"set multiplot\n");
//    fprintf(fp,"set cbrange [0.5:3.5]\n");
//    fprintf(fp,"set palette maxcolors 3\n");
//    fprintf(fp,"set view 0, 90, 1, 1\n");
//    fprintf(fp,"set palette defined ( 1 '#ff6969', 2 '#44d37e', 3 '#6ec4ff')\n");
//    fprintf(fp,"set cblabel \"Y\"\n");
//
//    ofs.open("tmp_data.dat", std::ios::out);
//    MatrixXd temp(data.rows(),3);
//    temp << data, Y;
//    ofs << temp;
//    ofs.close();
//
//    fprintf(fp,"set nokey\n");
//    fprintf(fp,"splot 'tmp_data.dat' with points palette pt 7 notitle \n");
//
//    fflush(fp);
//    pclose(fp);
//
//    return true;
//}

bool plot(const MatrixXd &data, const VectorXd &Y) {

    if(data.cols() != 2) {
        cerr << "data should be two-dimension" << endl;
        return false;
    }

    FILE *fp = popen("gnuplot", "w");
    if (fp == NULL)
        return false;

    ofstream ofs;

    fprintf(fp,"set terminal postscript eps enhanced color\n");
    fprintf(fp,"set output 'plot.eps'\n");
    fprintf(fp,"set multiplot\n");
    fprintf(fp,"set cbrange [0.5:3.5]\n");
    fprintf(fp,"set palette maxcolors 3\n");
    fprintf(fp,"set palette defined ( 0 '#ff6969', 1 '#44d37e', 2 '#6ec4ff')\n");
    fprintf(fp,"set cblabel \"Y\"\n");

    ofs.open("tmp_data.dat", std::ios::out);
    MatrixXd temp(data.rows(),3);
    temp << data, Y;
    ofs << temp;
    ofs.close();

    fprintf(fp,"set nokey\n");
    fprintf(fp,"plot 'tmp_data.dat' with points palette pt 7 notitle \n");

    fflush(fp);
    pclose(fp);

    return true;
}
