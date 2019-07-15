
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include<tuple>
#include<map>
#include<fstream>


using namespace std;

struct network
{
	int n_x, n_h, n_y;
};

typedef struct network Struct;
const int N = 200;

class NetworkStructure
{
public:
	int n_x, n_h, n_y;

	NetworkStructure(int x, int h, int y)
	{
		n_x = x;
		n_h = h;
		n_y = y;
	}
};

class Parameters
{
public:
	vector<vector<float>> w1, w2, b1, b2;

	Parameters(vector<vector<float>> w1, vector<vector<float>> w2, vector<vector<float>> b1, vector<vector<float>> b2)
	{
		this->w1 = w1;
		this->w2 = w2;
		this->b1 = b1;
		this->b2 = b2;
	}
};
//Cache for z1,z2,a1,a2
class Cache
{
public:
    vector<vector<float>> z1, z2, a1, a2;

    Cache(vector<vector<float>> z1,vector<vector<float>> z2,vector<vector<float>> a1, vector<vector<float>> a2)
    {
        this->z1 = z1;
        this->z2 = z2;
        this->a1 = a1;
        this->a2 = a2;
    }
};

class Grads
{
public:
    vector<vector<float>> dw1, dw2, db1, db2;

    Grads(vector<vector<float>> dw1,vector<vector<float>> dw2,vector<vector<float>> db1, vector<vector<float>> db2)
    {
        this->dw1 = dw1;
        this->dw2 = dw2;
        this->db1 = db1;
        this->db2 = db2;
    }
};

float sigmoid(float a)
{
	float s = 1 / (1 + exp(-a));
	return s;
}

float **matrixDot(vector<vector<float> > a1, vector<vector<float> > a2)
{
	int a1R = a1.size();
	int a2C = a2[0].size();
	int a1C = a1[0].size();
	float **c = new float *[a1R];

	for (int i = 0; i < a1R; i++)
	{
		c[i] = new float[a2C];
		for (int j = 0; j < a2C; j++)
		{
			c[i][j] = 0;
			for (int k = 0; k < a1C; k++)
			{
				c[i][j] = a1[i][k] * a2[k][j];
			}
		}
	}
	return c;
}

//Initialise network structure which contains dimensions of input, hidden and output layer
NetworkStructure network_structure(vector<vector<float>> X, vector<vector<float>> Y)
{
	return NetworkStructure(X.size(), 4, Y.size());
}

Parameters parameter_init(NetworkStructure n1)
{
	vector<vector<float>> w1(n1.n_h), w2(n1.n_y), b1(n1.n_h), b2(n1.n_y);

	//for initializing w1
	for (int i = 0; i < n1.n_h; i++)
	{
		for (int j = 0; j < n1.n_x; j++)
		{
			w1[i].push_back((rand() % 6 + 2)*0.01);
		}
	}

	//for initializing w2
	for (int i = 0; i < n1.n_y; i++)
	{
		for (int j = 0; j < n1.n_h; j++)
		{
			w2[i].push_back((rand() % 6 + 2)*0.01);
		}
	}

	//for initializing b1
	for (int i = 0; i < n1.n_h; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			b1[i].push_back(0);
		}
	}
	//for initializing b2
	for (int i = 0; i < n1.n_y; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			b2[i].push_back(0);
		}
	}
	Parameters parameters(w1, w2, b1, b2);

	return parameters;
}


Cache forward_prop(vector<vector<float> > X, Parameters parameters) {

    vector<vector<float>> &w1 = parameters.w1;
    vector<vector<float>> &w2 = parameters.w2;
    vector<vector<float>> &b1 = parameters.b1;
    vector<vector<float>> &b2 = parameters.b2;
    int n1=w1.size();
    int n2=X[0].size();
    int n3=X.size();
    int n4=w2.size();
    vector<vector<float>> z1(n1),z2(n4),a1(n1),a2(n4);
   // cout << w1[0][0];

    for(int i=0;i<n1;i++)
    {
        for(int j=0;j<n2;j++)
        {
            z1[i].push_back(0);
            //cout << z1[i][j];
            for(int k=0;k<n3;k++)
            {
                z1[i][j]+=(w1[i][k]*X[k][j]);
            }
            z1[i][j] += b1[i][j];
            a1[i].push_back(sigmoid(z1[i][j]));
            //cout << a1[i][j];
        }
    }
/*
    for(int i=0;i<z1.size();i++){
        for(int j=0;j<z1[i].size();j++){
            a1[i].push_back(sigmoid(z1[i][j]));
        }
    }
*/

    for(int i=0;i<w2.size();i++)
    {
        for(int j=0;j<a1[i].size();j++)
        {
            z2[i].push_back(0);
            for(int k=0;k<w2[i].size();k++)
            {
                z2[i][j] += (w2[i][k]*a1[k][j]);
            }
            z2[i][j] += b2[i][j];
            //cout<<z2[i][j];
            a2[i].push_back(sigmoid(z2[i][j]));
            //cout << a2[i][j];
        }
    }

/*
    for(int i=0;i<z2.size();i++){
        for(int j=0;j<z2[i].size();j++){
            a2[i].push_back(sigmoid(z2[i][j]));
        }
    }
*/
    Cache cache(z1,z2,a1,a2);

    return cache;

}

float prediciton_cost(vector<vector<float>> a2, vector<vector<float>> Y)
{
    float m = Y.size();
    vector<vector<float>> cross_entropy(a2.size());
    float cost=0.0;
    for(int i=0;i<a2.size();i++)
    {
        for(int j=0;j<a2[i].size();j++)
        {
            cross_entropy[i].push_back((log(a2[i][j])*Y[i][j]) + (log(1-a2[i][j])*(1-Y[i][j])));
            cost += cross_entropy[i][j];
        }
    }
    cost = cost / m;
    return cost;
}

Parameters backward_prop(Parameters parameters, Cache cache, vector<vector<float>> X, vector<vector<float>> Y,NetworkStructure n1)
{
    int m = X[0].size();
    //Retrieving W1 and W2
    vector<vector<float>> &w1 = parameters.w1;
    vector<vector<float>> &w2 = parameters.w2;
    vector<vector<float>> &b1 = parameters.b1;
    vector<vector<float>> &b2 = parameters.b2;
    //Retrieving A1 and A2
    vector<vector<float>> &a1 = cache.a1;
    vector<vector<float>> &a2 = cache.a2;
    int n=w1.size();
    int n2=X[0].size();
    int n3=X.size();
    int n4=w2.size();

    vector<vector<float>> dz1(n),dz2(n4),dw1(n1.n_h),db1(n1.n_h),db2(n1.n_y),a1T(a1[0].size()),w1T(n3),XT(n2);

    float sum=0.0;
    int x=Y.size(),y=Y[0].size();
    //for dz2 and db2
    for(int i=0;i<x;i++){
        for(int j=0;j<y;j++){
                dz2[i].push_back(a2[i][j] - Y[i][j]);
                db2[i].push_back(dz2[i][j] / m);
        }
    }

    //for a1-Transpose
    for(int i=0;i<a1.size();i++){
        for(int j=0;j<a1[i].size();j++){
                a1T[j].push_back(a1[i][j]);
        }
    }

    //for w1-transpose
    for(int i=0;i<w1.size();i++){
        for(int j=0;j<w1[i].size();j++){
                w1T[j].push_back(w1[i][j]);
        }
    }

    //for X-Transpose
    for(int i=0;i<X.size();i++){
        for(int j=0;j<X[i].size();j++){
                XT[j].push_back(X[i][j]);
        }
    }

    //for dw2
    vector<vector<float>> dw2(n1.n_y, vector<float>(n1.n_h));
    for(int i=0;i<dz2.size();i++)
    {
        for(int j=0;j<a1T[0].size();j++)
        {
            dw2[i].push_back(0);
            for(int k=0;k<dz2[0].size();k++)
            {
                dw2[i][j] += (dz2[i][k]*a1T[k][j]);
            }
            dw2[i][j] = dw2[i][j]/m;
        }
    }

    //or dz1 and db1
    vector<vector<float>> temp(n3,vector<float>(dz2[0].size()));
    for(int i=0;i<w1T.size();i++)
    {
        for(int j=0;j<dz2[0].size();j++)
        {
            temp[i].push_back(0);
            for(int k=0;k<dz2.size();k++)
            {
                temp[i][j] += w1T[i][k] * dz2[k][j];
            }
        }
    }

    for(int i=0;i<temp.size();i++){
        for(int j=0;j<temp[i].size();j++){
            dz1[i].push_back(temp[i][j] * (1-pow(a1[i][j],2)));
            db1[i].push_back(dz1[i][j]/m);
        }
    }

    //for dw1
    for(int i=0;i<dz1.size();i++){
        for(int j=0;j<XT[0].size();j++){
            dw1[i].push_back(0);
            for(int k=0;k<XT.size();k++){
                dw1[i][j] += dz1[i][k] + XT[k][j];
            }
            dw1[i][j] /= m;
        }
    }

    //updating parameters

    float l = 0.01;
    for (int i = 0; i < n1.n_h; i++)
	{
		for (int j = 0; j < n1.n_x; j++)
		{
			w1[i][j] = w1[i][j] - l*dw1[i][j];
		}
	}

	//for initializing w2
	for (int i = 0; i < n1.n_y; i++)
	{
		for (int j = 0; j < n1.n_h; j++)
		{
			w2[i][j] = w2[i][j] - l*dw2[i][j];
		}
	}

	//for initializing b1
	for (int i = 0; i < n1.n_h; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			b1[i][j] = b1[i][j] - l*db1[i][j];
		}
	}

	//for initializing b2
	for (int i = 0; i < n1.n_y; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			b2[i][j] = b2[i][j] - l*db2[i][j];
		}
	}


	return parameters;

   // Grads grads(dw1,dw2,db1,db2);

  //  return grads;



}



int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
void ReadMNIST(int NumberOfImages, int DataOfAnImage,vector<vector<double>> &arr)
{
    arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
    ifstream file ("C:\\Users\\Shail\\Documents\\Remote sensing\\Data\\train-images-idx3-ubyte",ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= (double)temp;

                }
                cout << arr[i][r];
            }
        }
    }
    else{
        cout << "hey else";
    }
}

int main()
{
	//Making the dataset
	vector<vector<float> > X = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12} };
	vector<vector<float> > Y = { {1,0,1,0} };
    float cost;


	//getting network structure
	//NetworkStructure s = network_structure(X, Y);

	//Parameters are initialised
	//Parameters parameters = parameter_init(s);
    //for(int i=0;i<10;i++){
	//forward propagation
    //Cache cache = forward_prop(X,parameters);

    //cost prediction
    /*cost = prediciton_cost(cache.a2,Y);
    cout << "cost " << i << " : ";
    cout << cost <<endl;

    parameters = backward_prop(parameters,cache,X,Y,s);
    }*/

    vector<vector<double>> ar;
    ReadMNIST(55000,784,ar);

	return 0;
}
