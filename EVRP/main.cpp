#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <vector>
#include <set>
#include <cmath>
#include <unordered_set>
#include <cmath>
#include <numeric>
#include <stdio.h>
#include <cstdio>
#include <stack>
#include <queue>
#include <fstream>
#include <sstream>
#include <map>
#include <random>

#define CHAR_LEN 100
#define TERMINATION 25000*ACTUAL_PROBLEM_SIZE  	//DO NOT CHANGE THE NUMBER
#define MAX_TRIALS 	20 									//DO NOT CHANGE THE NUMBER
#define CHAR_LEN 100

using namespace std;


//Constant Genetice Algorithm
const float MUTATION_RATE = 0.8;
const int POPULATION_SIZE = 200;
const int POOLING_SIZE = 50;
const int GENERATION = 200;

const double eps = 1e-6;
/****************************************************************/
/*                Main Function                                 */
/****************************************************************/

struct solution {
	// This doesn't contain stations
	vector<int> tour;
	int id;
	double tour_length;
	bool operator<(const struct solution& a)const {
		return tour_length < a.tour_length;
	}
	bool operator>(const struct solution& a)const {
		return tour_length > a.tour_length;
	}
};

struct solutionFull {
	// This contains station
	vector<int> tour;
	int id;
	double tour_length;
	int steps;
};

struct node {
	int id;
	double x;
	double y;
};

struct point
{
	int id;
	double x;
	double y;
	bool add = false;
};

char* problem_instance;
vector<struct node> node_list;			//List of nodes with id and x and y coordinates
vector<int> cust_demand;                //List with id and customer demands
vector<bool> charging_station;			//Whether id i is a charging station
vector<vector<double>> distances;       //Distance matrix
int problem_size;						//Problem dimension read
vector<vector<int>>cluster;				//nCluster=MIN_VEHICLES
double energy_consumption;

int NUM_OF_CUSTOMERS;			//number of customer set
int ACTUAL_PROBLEM_SIZE; 		//total number of nodes
int NUM_OF_STATIONS;			//number of charging stations
int MAX_CAPACITY;				//maxmimum cargo capacity
int DEPOT;						//id of the depot
double OPTIMUM;					//Global optimum (or upper bound) of the problem instance (if known)
int BATTERY_CAPACITY;			//maximum energy level
int MIN_VEHICLES;				//minimum number of cluster

FILE* log_performance;
char* perf_filename;
double* perf_of_trials;
//------------------------------------------

void read_problem(char* filename);					//reads .evrp file 
double get_energy_consumption(int from, int to);	//returns the energy consumption 
int get_customer_demand(int customer);				//returns the customer demand
double get_distance(int from, int to);				//returns the distance
bool is_charging_station(int node);					//returns true if node is a charging station
void getCluster();									//init random cluster, result will affect on global var "cluster"
void getCluster1();
void stabilize();
// 2opt LS													
void do2Opt(vector<int>&, int, int);
void LS(vector<vector<int>>&);
// 3opt LS
void LS3(vector<vector<int>>&);
vector<pair<int, pair<int, int>>>all_segment(int);
double reverse_segment(vector<int>&, int, int, int);

vector<solution>initialize(int);													//Init random population (size of 2 at the beginning)
vector<int> selection(vector<double>&, int);
vector<int> selectiontopk(vector<double>&, int);								//Selection
solution hsm(solution);
solution hmm(solution);
void filling_population(vector<solution>&, bool);
solutionFull geneticAlgo();														//Genetic Algo
solutionFull AddStation(solution&);												//Adding stations to a chromosome
solutionFull AddStation1(solution&);
vector<int>convert_chromosome(vector<vector<int>>&);							//Convert a chromosome to a cluster
vector<vector<int>>convert_cluster(solution&);									//Convert a cluster to a solution (without station)
pair<solution, solution> crossover_operation(const solution&, const solution&);	//Crossover operation

pair<long long, long long>hashh(vector<int>&);
vector<int> FindStation(int, int, double, vector<int>);

int main(int argc, char* argv[])
{
	int run;

	problem_instance = argv[1];     // pass the .evrp filename as an argument
	read_problem(problem_instance);
	geneticAlgo();

	//int num = 50;
	//vector<double>result;
	//vector<double>result1;
	//for (int step = 1; step <= 50; step++)
	//{
	//	cout << step << "\n";
	//	// First init solution
	//	getCluster();

	//	vector<vector<int>>tmp=cluster;
	//	//LS(cluster);

	//	solution first;
	//	first.id = step;
	//	// Calculate its fitness
	//	first.tour = convert_chromosome(cluster);
	//	double ans = 0;
	//	for (int i = 0; i <= (first.tour.size() - 2); i++)
	//		ans = ans + get_distance(first.tour[i], first.tour[i + 1]);
	//	result.push_back(ans);

	//	cluster = tmp;
	//	getCluster1();
	//	//LS(cluster);

	//	first.tour = convert_chromosome(cluster);
	//	ans = 0;
	//	for (int i = 0; i <= (first.tour.size() - 2); i++)
	//		ans = ans + get_distance(first.tour[i], first.tour[i + 1]);
	//	result1.push_back(ans);
	//}
	//for (auto v : result)cout << setprecision(8) << v << " ";
	//cout << "\n";
	//for (auto v : result1)cout << setprecision(8) << v << " ";
	//cout << "\n";
	//double s = 0;
	//for (auto v : result)s += v;
	//s /= num;
	//cout << setprecision(8) << s;
	//cout << " ";
	//s = 0;
	//for (auto v : result1)s += v;
	//s /= num;
	//cout << setprecision(8) << s;
	return 0;
}
/*
	Convex Clustering
*/
vector<point> ans;
double Euclideandistance(point a, point b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
};

void getAns(vector<vector<point>> cvHulls, int index, int id)
{
	if (cvHulls[index].size() <= 2)
	{
		while (index != cvHulls.size())
		{
			for (auto theRest : cvHulls[index])
				ans.push_back(theRest);
			index++;
		}
		return;
	}

	int start = -1;
	for (int i = 0; i < cvHulls[index].size(); i++)
		if (cvHulls[index][i].id == id)
		{
			start = i;
			break;
		}

	ans.push_back(cvHulls[index][start]);

	int i;
	if (start == 0)
	{
		i = cvHulls[index].size() - 2;
		for (; i != 0; i--)
		{
			if (cvHulls[index][i].add)
			{
				getAns(cvHulls, index + 1, cvHulls[index][i].id);
			}
			else
			{
				ans.push_back(cvHulls[index][i]);
			}
		}
	}
	else
	{
		i = start - 1;
		for (; i != start; i--)
		{
			if (cvHulls[index][i].add)
			{
				getAns(cvHulls, index + 1, cvHulls[index][i].id);
			}
			else
			{
				ans.push_back(cvHulls[index][i]);
			}

			if (i - 1 == -1)
			{
				i = cvHulls[index].size() - 1;
			}
		}
	}
}

void hull(vector<vector<point>> convexHulls)
{
	ans.clear();
	/*vt<vt<point>> = {
		{ {1, 0, 0}, {2,3, 0}, {3,4, 4}, {4,-4, 4}, {5,-3, 2}, {1, 0, 0}},
		{ {6, 0, 1}, {7, 1.5, 2.5}, {8,0, 3.5}, {9, -1, 2}, {6, 0, 1} }
	};*/

	long long nConvexHull = convexHulls.size();
	for (int o = 0; o < nConvexHull - 1; o++)
	{
		if (convexHulls[o].size() <= 2)
			continue;
		bool checkBreak = false;
		for (int op = 0; op < convexHulls[o].size() - 1; op++)
		{
			point currentOp = convexHulls[o][op];
			point neighborOp = convexHulls[o][op + 1];

			bool flag = false;

			if (convexHulls[o + 1].size() <= 2)
			{
				point temp = convexHulls[o + 1][0];
				temp.add = true;
				convexHulls[o].insert(convexHulls[o].begin() + op + 1, temp);
				flag = true;
				checkBreak = true;
			}
			else
			{
				for (int i = 0; i < convexHulls[o + 1].size() - 1; i++)
				{
					point currentCheck = convexHulls[o + 1][i];
					point neighborCheck = convexHulls[o + 1][i + 1];
					if (Euclideandistance(currentOp, currentCheck) + Euclideandistance(neighborOp, neighborCheck) <
						Euclideandistance(currentOp, neighborOp) + Euclideandistance(currentCheck, neighborCheck))
					{
						currentCheck.add = true;
						convexHulls[o].insert(convexHulls[o].begin() + op + 1, currentCheck);
						flag = true;
						checkBreak = true;
						break;
					}
				}
			}

			if (flag)
				break;
		}
		if (!checkBreak)
		{
			point currentCheck = convexHulls[o + 1][0];
			currentCheck.add = true;
			convexHulls[o].insert(convexHulls[o].begin() + 1, currentCheck);
		}
	}

	for (int i = 0; i < convexHulls[0].size(); i++)
	{
		if (!convexHulls[0][i].add)
			ans.push_back(convexHulls[0][i]);
		else
			getAns(convexHulls, 1, convexHulls[0][i].id);
	}
}

//Kiet

point p0;

point pointNextToTop(vector<point>& st)
{
	point point_replace = st.back();
	st.pop_back();

	point point_res = st.back();
	st.push_back(point_replace);

	return point_res;
}

/*
	Square of distance between point1 & point2
*/
double distance2points(point point1, point point2)
{
	return (point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y);
}

/*
	 Find orientation of ordered triplet (point1, point2, point3)
	 0 --> point1, point2 and point3 are collinear
	 1 --> point1, point2 and point3 are clockwise
	 -1 --> point1, point2 and point3 are counterclockwise
*/
int orientation3points(point point1, point point2, point point3)
{
	double value_orientation = (point2.y - point1.y) * (point3.x - point2.x) - (point2.x - point1.x) * (point3.y - point2.y);
	if (value_orientation > 0)
		return 1;
	else if (value_orientation < 0)
		return -1;
	return 0;
}

// used for sorting points according to polar order w.r.t the pivot
bool POLAR_ORDER(point a, point b) {
	int order = orientation3points(p0, a, b);
	if (order == 0)
		return distance2points(p0, a) >= distance2points(p0, b);
	return (order == 1);
}

/*
	Implementation Graham Scan algorithm in order to find the convex hull
*/

bool CompareDoubles2(double A, double B)
{
	double diff = A - B;
	return (diff < eps) && (-diff < eps);
}

vector<point> GrahamScan(vector<point>& cluster)
{
	// Size of cluster
	int n = cluster.size();
	double yMin = cluster[0].y;
	int idxMin = 0;

	// Pick the bottom - most or chose the left most point in case of tie
	for (int i = 1; i < n; i++)
	{
		if ((cluster[i].y < yMin) || (CompareDoubles2(yMin, cluster[i].y) && cluster[i].x < cluster[idxMin].x))
		{
			yMin = cluster[i].y;
			idxMin = i;
		}
	}

	// Place the bottom-most point at first position
	swap(cluster[0], cluster[idxMin]);

	// Sort n-1 points with respect to ther first point
	p0 = cluster[0];
	//qsort(&cluster[1], n - 1, sizeof(cluster), compare);
	sort(cluster.begin() + 1, cluster.end(), POLAR_ORDER);
	vector<point>st;

	if (cluster.size() < 3)
		return cluster;
	else
	{
		st.push_back(cluster[0]);
		st.push_back(cluster[1]);
		st.push_back(cluster[2]);

		for (int i = 3; i < n; i++)
		{
			while (st.size() > 1 && orientation3points(pointNextToTop(st), st.back(), cluster[i]) == -1)
				st.pop_back();
			st.push_back(cluster[i]);
		}
		return st;
	}
}

void removepoint(vector<point>& points, int idd)
{
	points.erase(
		remove_if(points.begin(), points.end(), [&](point const& point) {
			return point.id == idd;
			}),
		points.end());
}

vector<vector<point>> multiConvexHull(vector<point>& cluster)
{
	vector<vector<point>>result;
	while (!cluster.empty())
	{
		vector<point>convex = GrahamScan(cluster);
		result.push_back(convex);

		while (!convex.empty())
		{
			removepoint(cluster, convex.back().id);
			convex.pop_back();
		}
	}
	return result;
}

//---------------------------------
/*
	Hash function
*/
pair<long long, long long>hashh(vector<int>& s)
{
	const long long p1 = 31;
	const long long m1 = (long long)1e9 + 7;
	long long pow1 = 1;
	const long long p2 = 53;
	const long long m2 = (long long)1e9 + 9;
	long long pow2 = 1;
	long long h1;
	long long h2;

	long long n = (long long)s.size();
	h1 = (s[0] * pow1) % m1;
	h2 = (s[0] * pow2) % m2;
	for (long long i = 1; i <= n - 1; i++)
	{
		pow1 *= p1;
		pow1 %= m1;
		pow2 *= p2;
		pow2 %= m2;
		h1 = (h1 + (pow1 * s[i]) % m1) % m1;
		h2 = (h2 + (pow2 * s[i]) % m2) % m2;
	}
	return { h1,h2 };
}
/*
* Genetic Algo
*/
set<pair<long long, long long>> kiemtra;
vector<solution>initialize(int number)
{
	vector<solution>result;
	for (int step = 0; step <= number - 1; step++)
	{
		// First init solution
		getCluster();
		getCluster1();
		//LS3(cluster);
		solution first;
		first.id = step;
		// Calculate its fitness
		first.tour = convert_chromosome(cluster);
		//for (int i = 0; i <= MIN_VEHICLES - 1; i++)
		//{
		//	cout << "Cluster " << i << ": ";
		//	for (auto v : cluster[i])cout << v << " ";
		//	cout << "\n";
		//}
		solutionFull firstFull = AddStation(first);
		result.push_back(first);
	}
	return result;
}
solutionFull geneticAlgo()
{

	// Random a pair of solutions Lac Long Quan vs Au Co
	vector<solution> ini = initialize(POPULATION_SIZE);

	// Min_Heap
	vector<solution>q = ini;

	double globalcost = 99999999.0;
	solution globalans;
	// Genetic
	int step = 0;
	int generation = GENERATION;
	vector<double>loss;
	while (step < generation)
	{

		for (int i = 0; i <= (int)q.size() - 1; i++)
			if (q[i].tour_length < globalcost)
				globalcost = q[i].tour_length, globalans = q[i];
		if (1)
		{
			solutionFull globalfull = AddStation(globalans);
			for (auto x : globalans.tour)cout << x << " ";
		}
		// Get POOLING_SIZE best solution
		loss.push_back(globalcost);
		vector<double>score;
		for (auto vv : q)
			score.push_back(vv.tour_length);
		int ski = 2 * GENERATION / 100;
		vector<solution>tmp;
		bool f = 0;
		if (step - ski >= 0 && CompareDoubles2(loss[step], loss[step - ski]))
		{
			vector<int>choose = selectiontopk(score, POOLING_SIZE / 2);
			set<int>tmps;
			for (auto v : choose)tmps.insert(v);
			for (int i = 0; i <= (int)q.size() - 1; i++)
				if (tmps.find(i) != tmps.end())
					tmp.push_back(q[i]);
			vector<solution>tam = initialize(POOLING_SIZE / 2);
			for (auto vv : tam)
				tmp.push_back(vv);
			f = 1;

		}
		else
		{
			vector<int>choose = selectiontopk(score, POOLING_SIZE);
			set<int>tmps;
			for (auto v : choose)tmps.insert(v);
			for (int i = 0; i <= (int)q.size() - 1; i++)
				if (tmps.find(i) != tmps.end())
					tmp.push_back(q[i]);
		}
		q = tmp;
		filling_population(q, f);

		double mi = 99999999.0;
		for (auto v : q)
			mi = min(mi, v.tour_length);
		cout << step << "/" << generation << "....Tour length: " << mi << "-----global: " << globalcost << "\n";
		step++;
	}

	// Get the best out of it
	cout << globalcost;
	solutionFull globalfull = AddStation(globalans);
	for (auto x : globalfull.tour)cout << x << " ";
	return globalfull;
}
void filling_population(vector<solution>& q, bool f)
{
	random_device rd;
	default_random_engine eng(rd());
	uniform_real_distribution<float> distr(0, 1);
	int dem = 0;
	set<pair<long long, long long>>s;
	for (auto cc : q)
		s.insert(hashh(cc.tour));
	while ((int)q.size() <= POPULATION_SIZE)
	{
		vector<double>score;
		for (auto vv : q)
			score.push_back(vv.tour_length);

		vector<int>choose = selection(score, 2);


		int id1 = choose[0];
		int id2 = choose[1];
		solution a = q[id1];
		solution b = q[id2];
		// Crossover
		pair<solution, solution> tmp = crossover_operation(a, b);

		// LS to improve its fitness

		cluster = convert_cluster(tmp.first);

		//LS(cluster);
		getCluster1();
		tmp.first.tour = convert_chromosome(cluster);


		cluster = convert_cluster(tmp.second);
		//LS(cluster);
		getCluster1();
		tmp.second.tour = convert_chromosome(cluster);
		double rate = distr(eng);
		if (rate < MUTATION_RATE || f)
		{
			double random = distr(eng);
			if (random < 0.5)
				tmp.first = hsm(tmp.first);
			else
				tmp.second = hmm(tmp.second);
		}


		// Updating its fitness value according to
		solutionFull tmpfirstFull = AddStation(tmp.first);
		solutionFull tmpsecondFull = AddStation(tmp.second);
		// If they're valid
		if (tmp.first.tour_length < 9999999.0 && s.find(hashh(tmp.first.tour)) == s.end())
			q.push_back(tmp.first), s.insert(hashh(tmp.first.tour));
		if (tmp.second.tour_length < 9999999.0 && s.find(hashh(tmp.second.tour)) == s.end())
			q.push_back(tmp.second), s.insert(hashh(tmp.second.tour));

	}
	while ((int)q.size() != POPULATION_SIZE)
	{
		q.pop_back();
	}

}
void stabilize()
{
	int num_cluster = (int)cluster.size();
	//Balancing
	int avg = 0;
	for (int i = 0; i <= num_cluster - 1; i++)
		avg += (int)cluster.size();
	avg /= num_cluster;

	for (int smallest = 0; smallest <= (int)cluster.size() - 1; smallest++)
	{
		while (cluster[smallest].size() < avg)
		{
			srand(time(0));
			int c = rand() % ((int)cluster[smallest].size());
			if (c == 0)c++;
			else if (c == (cluster[smallest].size() - 1))c--;
			int A = cluster[smallest][c];
			vector<pair<double, pair<int, int>>>pp;
			for (int i = 0; i <= num_cluster - 1; i++)
				if (i != smallest)
					for (auto cc : cluster[i])pp.push_back({ get_distance(cc,A),{cc,i} });
			sort(pp.begin(), pp.end());

			int s = 0;
			for (auto cc : cluster[smallest])s += get_customer_demand(cc);

			bool f = 0;
			for (auto xx : pp)
			{
				int B = xx.second.first;
				int route = xx.second.second;
				int calB = 0;
				int calA = 0;
				for (auto cc : cluster[route])calB += get_customer_demand(cc);
				for (auto cc : cluster[smallest])calA += get_customer_demand(cc);
				int delta = abs(calA + get_customer_demand(B) - (calB - get_customer_demand(B)));
				if (get_customer_demand(B) + s<MAX_CAPACITY && abs(calA - calB)>delta)
				{
					cluster[smallest].push_back(B);
					s += get_customer_demand(B);
					int idx = 0;
					for (int i = 0; i <= (int)cluster[route].size() - 1; i++)
						if (cluster[route][i] == B)
							idx = i;
					cluster[route].erase(cluster[route].begin() + idx);
					f = 1;
					break;
				}
			}
			if (!f)break;
		}
	}
}
/*
	Natural Selection_ Roullete wheel
*/
vector<int> selection(vector<double>& fitness, int num_selected)
{
	random_device rd;
	default_random_engine eng(rd());
	uniform_real_distribution<float> distr(0, 1);

	// Max element in fitness
	double max_fitness = *max_element(fitness.begin(), fitness.end());

	// Sum fitnesses of all individuals in population
	double sum_fitness = accumulate(fitness.begin(), fitness.end(), 0);

	// The Probability for each of individual in population
	vector<float>prob;
	for (auto& fit : fitness)
		prob.push_back((double)((double)max_fitness - fit + 1.0) / (((double)max_fitness + 1) * fitness.size() - sum_fitness));

	// Caculate the cumulative probability for each individual use Prefix array
	vector<pair<double, int>>cumulative_prob;
	cumulative_prob.push_back(make_pair(prob[0], 0));
	for (int i = 1; i < prob.size(); i++)
		cumulative_prob.push_back(make_pair(cumulative_prob[i - 1].first + prob[i], i));

	// switch reselect 
	bool reselect = false;

	vector<int>res;
	for (int i = 1; i <= num_selected; i++)
	{
		double r = distr(eng);

		if (r < cumulative_prob[0].first)
		{
			res.push_back((*cumulative_prob.begin()).second);
			if (!reselect)
				cumulative_prob.erase(cumulative_prob.begin());
		}
		else if (r > cumulative_prob.back().first)
		{
			res.push_back(cumulative_prob.back().second);
			if (!reselect)
				cumulative_prob.pop_back();
		}
		else
		{
			for (int j = 0; j < cumulative_prob.size(); j++)
			{
				if (cumulative_prob[j].first > r)
				{
					res.push_back(cumulative_prob[j].second);
					if (!reselect)
						cumulative_prob.erase(cumulative_prob.begin() + j);
					break;
				}
			}
		}
	}
	return res;
}

vector<int> selectiontopk(vector<double>& fitness, int num_selected)
{
	vector<int>res;
	vector<pair<double, int>>p;
	for (int i = 0; i <= (int)fitness.size() - 1; i++)p.push_back({ fitness[i],i });
	sort(p.begin(), p.end());
	for (int i = 0; i <= (num_selected - 1); i++)
		res.push_back(p[i].second);

	return res;
}
/*
	Mutation
*/
solution hsm(solution sol)
{
	solution result = sol;
	int n = sol.tour.size();
	int cus1Idx = 0;
	srand(time(0));
	while (sol.tour[cus1Idx] == 0)
		cus1Idx = rand() % n;

	vector<int> flag(n, 0);
	for (int i = cus1Idx; i < n && sol.tour[i] != 0; i++)
		flag[i] = 1;

	for (int i = cus1Idx; i >= 0 && sol.tour[i] != 0; i--)
		flag[i] = 1;

	double distance = 99999999999.0;
	int cus2Idx = 0;
	for (int i = 0; i < n; i++)
	{
		if (flag[i] == 1 || sol.tour[i] == 0)
			continue;
		if (distances[result.tour[i]][result.tour[cus1Idx]] < distance)
		{
			distance = distances[result.tour[i]][result.tour[cus1Idx]];
			cus2Idx = i;
		}
	}
	swap(result.tour[cus1Idx], result.tour[cus2Idx]);

	return result;
}
solution hmm(solution sol)
{
	solution result = sol;
	int n = sol.tour.size();
	int cus1Idx = 0;
	srand(time(0));
	while (sol.tour[cus1Idx] == 0)
		cus1Idx = rand() % n;

	vector<int> flag(n, 0);
	for (int i = cus1Idx; i < n && sol.tour[i] != 0; i++)
		flag[i] = 1;

	for (int i = cus1Idx; i >= 0 && sol.tour[i] != 0; i--)
		flag[i] = 1;

	double distance = 99999999999.0;
	int cus2Idx = 0;
	for (int i = 0; i < n; i++)
	{
		if (flag[i] == 1 || sol.tour[i] == 0)
			continue;
		if (distances[result.tour[i]][result.tour[cus1Idx]] < distance)
		{
			distance = distances[result.tour[i]][result.tour[cus1Idx]];
			cus2Idx = i;
		}
	}

	int cus2 = result.tour[cus2Idx];
	result.tour.insert(result.tour.begin() + cus1Idx, 1, cus2);
	if (cus2Idx > cus1Idx)
		result.tour.erase(result.tour.begin() + cus2Idx + 1);
	else
		result.tour.erase(result.tour.begin() + cus2Idx);

	return result;
}
/*
	Adding stations to solutions and map it to solutionFull
*/
solutionFull AddStation(solution& orig_sol)
{
	// NOTE: this can be used to calculate a fitness value
	/*
		INPUT: a solution not containing stations (chromosome)
		OUTPUT:  a solution containing stations

		Note: After calling this function, fitness value will be updated automatically to the solution &tmp
		Consider this is a calculating fitness function
	*/
	vector<int> stations;
	for (int i = 1; i < ACTUAL_PROBLEM_SIZE; i++)
		if (is_charging_station(i))
			stations.push_back(i);

	solutionFull res;
	res.id = orig_sol.id;
	res.tour.push_back(orig_sol.tour[0]);
	double e_tmp = BATTERY_CAPACITY;
	double c_tmp = MAX_CAPACITY;
	for (int i = 1; i < orig_sol.tour.size(); i++)
	{
		int from = res.tour[res.tour.size() - 1];
		int to = orig_sol.tour[i];
		c_tmp -= get_customer_demand(to);
		e_tmp -= get_energy_consumption(from, to);
		if (c_tmp < 0.0) {
			res.tour_length = orig_sol.tour_length = 99999999.0;
			return res;
		}
		if (e_tmp < 0.0)
		{
			e_tmp += get_energy_consumption(from, to);
			vector<int> add_stations = FindStation(from, to, e_tmp, stations);
			if (add_stations.empty() && is_charging_station(from))
			{
				res.tour_length = orig_sol.tour_length = 99999999.0;
				return res;
			}
			int btr = 1;
			while (add_stations.empty() && !is_charging_station(res.tour[res.tour.size() - btr - 1]))
			{
				btr++;
				int from_tmp = res.tour[res.tour.size() - btr];
				int to_tmp = res.tour[res.tour.size() - btr + 1];
				e_tmp += get_energy_consumption(from_tmp, to_tmp);
				add_stations = FindStation(from_tmp, to_tmp, e_tmp, stations);
			}
			if (add_stations.empty())
			{
				res.tour_length = orig_sol.tour_length = 99999999.0;
				return res;
			}
			else
			{
				if (add_stations.size() == 1)
				{
					res.tour.push_back(to);
					res.tour.insert(res.tour.end() - btr, 1, add_stations[0]);
				}
				else
				{
					res.tour.push_back(to);
					res.tour.insert(res.tour.end() - btr, 1, add_stations[0]);
					res.tour.insert(res.tour.end() - btr, 1, add_stations[1]);
				}
				e_tmp = BATTERY_CAPACITY;
				for (int j = res.tour.size() - btr - 1; j < res.tour.size() - 1; j++)
					e_tmp -= get_energy_consumption(res.tour[j], res.tour[j + 1]);
			}
		}
		else
			res.tour.push_back(to);
		if (to == DEPOT)
		{
			e_tmp = BATTERY_CAPACITY;
			c_tmp = MAX_CAPACITY;
		}
	}
	res.tour_length = 0;
	for (int i = 0; i < res.tour.size() - 1; i++)
		res.tour_length += get_distance(res.tour[i], res.tour[i + 1]);
	orig_sol.tour_length = res.tour_length;
	solution tmp = orig_sol;

	solutionFull res1 = AddStation1(tmp);
	orig_sol.tour_length = min(orig_sol.tour_length, res1.tour_length);

	if (res.tour_length < res1.tour_length)
		return res;
	else return res1;
}

vector<int> FindStation(int from, int to, double e_tmp, vector<int> stations)
{
	vector<int> stations1;
	vector<int> stations2;
	for (int station : stations)
	{
		if (get_energy_consumption(from, station) <= e_tmp)
			stations1.push_back(station);
		if (get_energy_consumption(station, to) <= BATTERY_CAPACITY)
			stations2.push_back(station);
	}
	vector<int> add_stations;
	if (stations1.size() == 0 || stations2.size() == 0)
		return add_stations;
	int s1 = -1;
	int s2 = -1;
	double e_flag = BATTERY_CAPACITY;
	for (int station2 : stations2)
	{
		double d = 9999999999.0;
		for (int station1 : stations1)
		{
			if (get_energy_consumption(station1, station2) <= BATTERY_CAPACITY && get_energy_consumption(station2, to) <= e_flag)
			{
				double d_tmp = get_distance(from, station1) + get_distance(station1, station2) + get_distance(station2, to);
				if (s2 != station2)
				{
					s1 = station1;
					s2 = station2;
					e_flag = get_energy_consumption(s2, to);
				}
				else if (d_tmp < d)
				{
					s1 = station1;
					s2 = station2;
					e_flag = get_energy_consumption(s2, to);
					d = d_tmp;
				}
			}
		}
	}
	if (s1 == -1)
		return add_stations;
	else if (s1 == s2)
		add_stations.push_back(s1);
	else
	{
		add_stations.push_back(s1);
		add_stations.push_back(s2);
	}
	return add_stations;
}

solutionFull AddStation1(solution& tmp)
{
	// NOTE: this can be used to calculate a fitness value
	/*
		INPUT: a solution not containing stations (chromosome)
		OUTPUT:  a solution containing stations

		Note: After calling this function, fitness value will be updated automatically to the solution &tmp
		Consider this is a calculating fitness function
	*/
	solutionFull res;
	res.tour_length = 0;
	res.id = tmp.id;
	double energy_temp = BATTERY_CAPACITY;
	double capacity_temp = MAX_CAPACITY;
	double distance_temp = 0.0;
	for (int i = 0; i <= (int)tmp.tour.size() - 2; i++)
	{
		int from = tmp.tour[i];
		int to = tmp.tour[i + 1];
		res.tour.push_back(tmp.tour[i]);
		capacity_temp -= get_customer_demand(to);
		energy_temp -= get_energy_consumption(from, to);
		distance_temp += get_distance(from, to);
		if (capacity_temp < 0.0) {
			res.tour_length = tmp.tour_length = 99999999.0;
			return res;
		}
		bool f = 0;
		if (energy_temp < 0.0) {
			int ans = -1;
			for (int j = 1; j <= ACTUAL_PROBLEM_SIZE - 1; j++)
				if ((ans == -1 && is_charging_station(j)) || (is_charging_station(j) && energy_temp + get_energy_consumption(from, to) - get_energy_consumption(from, j) >= 0 && get_distance(from, j) + get_distance(j, to) < get_distance(from, ans) + get_distance(ans, to)))
					ans = j;
			res.tour.push_back(ans);
			res.tour_length += get_distance(from, ans) + get_distance(ans, to);
			energy_temp = BATTERY_CAPACITY - get_energy_consumption(ans, to);
			f = 1;
		}
		if (to == DEPOT) {
			capacity_temp = MAX_CAPACITY, energy_temp = BATTERY_CAPACITY;
		}
		if (!f)
			res.tour_length += get_distance(from, to);
	}
	res.tour.push_back(DEPOT);
	double ans = 0;
	for (int i = 0; i <= (int)res.tour.size() - 2; i++)
		ans += get_distance(res.tour[i], res.tour[i + 1]);
	res.tour_length = ans;
	tmp.tour_length = ans;
	// Assigning its fitness (solutionFull) to solution
	return res;
}

/*
	Crossover operator
*/
pair<solution, solution> crossover_operation(const solution& parent1, const solution& parent2)
{
	/*
		INPUT: parent1 as type solution, paren2 as type solution
		OUTPUT: pair of tow solution as two children
	*/

	// Get list of distinct custumer
	set<int> setNodeId;
	for (auto x : parent1.tour)
		if (x != 0)
			setNodeId.insert(x);
	for (auto x : parent2.tour)
		if (x != 0)
			setNodeId.insert(x);

	// Choose a random customer index
	srand(time(0));
	int customerRandomIndex = rand() % setNodeId.size();
	if (customerRandomIndex == 0)customerRandomIndex++;
	set<int>::iterator it = setNodeId.begin();
	advance(it, customerRandomIndex);
	int customerRandom = *it;
	// Get route contain customerRandom in pr1 and pr2;
	solution V1, V2;
	// variable contain (V1 U V2);
	set<int> V1V2;
	V1V2.insert(customerRandom);

	for (int i = 0; i < parent1.tour.size(); i++)
	{
		if (parent1.tour[i] == customerRandom)
		{
			int left = i - 1, right = i + 1;
			for (; right < parent1.tour.size() && parent1.tour[right] != 0; right++)
			{
				V1V2.insert(parent1.tour[right]);
			}
			for (; left >= 0 && parent1.tour[left] != 0; left--)
			{
				V1V2.insert(parent1.tour[left]);
			}
			if (left < 0)
				left = 0;
			if (right == parent1.tour.size())
				right--;
			for (int j = left; j <= right; j++)
				if (parent1.tour[j] != 0)
					V1.tour.push_back(parent1.tour[j]);
			break;
		}
	}
	for (int i = 0; i < parent2.tour.size(); i++)
	{
		if (parent2.tour[i] == customerRandom)
		{
			int left = i - 1, right = i + 1;
			for (; right < parent2.tour.size() && parent2.tour[right] != 0; right++)
			{
				V1V2.insert(parent2.tour[right]);
			}
			for (; left >= 0 && parent2.tour[left] != 0; left--)
			{
				V1V2.insert(parent2.tour[left]);
			}
			if (left < 0)
				left = 0;
			if (right == parent2.tour.size())
				right--;
			for (int j = left; j <= right; j++)
				if (parent2.tour[j] != 0)
					V2.tour.push_back(parent2.tour[j]);
			break;
		}
	}

	// sub2 = (V1 U V2) \ V1;
	vector<int> sub1 = V1.tour;
	vector<int> sub2;
	for (auto x : V1V2)
	{
		if (find(sub1.begin(), sub1.end(), x) == sub1.end())
		{
			sub2.push_back(x);
		}
	}

	// SC1 = (sub2, sub1);
	// SC2 = (rsub1, rsub2);
	vector<int> SC1 = sub2, SC2(sub1.rbegin(), sub1.rend()), rsub2(sub2.rbegin(), sub2.rend());
	SC1.insert(SC1.end(), sub1.begin(), sub1.end());
	SC2.insert(SC2.end(), rsub2.begin(), rsub2.end());

	solution child1 = parent1, child2 = parent2;
	for (int i = 0, j = 0; i < child1.tour.size(); i++)
	{
		if (find(SC1.begin(), SC1.end(), child1.tour[i]) != SC1.end())
		{
			child1.tour[i] = SC1[j++];
			if (SC1.size() == j)
				break;
		}
	}
	for (int i = 0, j = 0; i < child2.tour.size(); i++)
	{
		if (find(SC2.begin(), SC2.end(), child2.tour[i]) != SC2.end())
		{
			child2.tour[i] = SC2[j++];
			if (SC2.size() == j)
				break;
		}
	}

	return make_pair(child1, child2);
}
/*
	Convert solution to cluster
*/
vector<vector<int>>convert_cluster(solution& t)
{
	/*
		INPUT: a solution
		OUTPUT: a cluster (vector<vector<int>>)
	*/
	vector<vector<int>>c;
	vector<int>path;
	for (int i = 0; i <= t.tour.size() - 1; i++)
	{

		if (t.tour[i] == DEPOT)
		{
			if (path.size() != 0)
			{
				path.push_back(DEPOT);
				c.push_back(path);
			}
			path.clear();
			path.push_back(DEPOT);
		}
		else
		{
			path.push_back(t.tour[i]);
		}
	}
	return c;
}
/*
	Convert a cluster (vector<vector<int>>) to chromosome (vector<int>)
*/
vector<int>convert_chromosome(vector<vector<int>>& g)
{
	/*
		INPUT: a cluster (vector<vector<int>>)
		OUTPUT: a chromosome (vector<int>)
	*/
	vector<int>chrom;
	for (auto v : g)
	{
		for (auto cc : v)
			chrom.push_back(cc);
		chrom.pop_back();
	}
	chrom.push_back(0);
	return chrom;
}
/*
	2 opt Local Search
*/
void do2Opt(vector<int>& path, int i, int j)
{
	reverse(begin(path) + i + 1, begin(path) + j + 1);
}
void LS(vector<vector<int>>& all_path)
{
	/*
		Input: a cluster
		Output: that cluster with modified cities
	*/
	for (int i = 0; i <= (int)all_path.size() - 1; i++)
	{
		//Erasing starting vertex at the end
		all_path[i].erase(all_path[i].begin() + (int)all_path[i].size() - 1);
		// The starting vertex is not included at the end
		vector<int> path = all_path[i];
		int n = path.size();
		bool foundImprovement = true;

		while (foundImprovement) {
			foundImprovement = false;

			for (int i = 0; i <= n - 2; i++) {
				for (int j = i + 1; j <= n - 1; j++) {
					int lengthDelta = -get_distance(path[i], path[i + 1 % n]) - get_distance(path[j], path[(j + 1) % n])
						+ get_distance(path[i], path[j]) + get_distance(path[i + 1 % n], path[(j + 1) % n]);

					// If the length of the path is reduced, do a 2-opt swap
					if (lengthDelta < 0) {
						do2Opt(path, i, j);
						foundImprovement = true;
					}
				}
			}
		}
		all_path[i] = path;
		all_path[i].insert(all_path[i].end(), 0);
	}
}
/*
	3 opt Local Search
*/
void LS3(vector<vector<int>>& all_path)
{
	for (int i = 0; i <= (int)all_path.size() - 1; i++)
	{
		vector<int>tour = all_path[i];
		tour.pop_back();
		while (1)
		{
			int delta = 0;
			vector<pair<int, pair<int, int>>>tmp = all_segment((int)tour.size());
			for (auto cc : tmp)
			{
				int a = cc.first;
				int b = cc.second.first;
				int c = cc.second.second;
				delta += reverse_segment(tour, a, b, c);
			}
			if (delta >= 0)
				break;
		}
		tour.push_back(0);
		all_path[i] = tour;
	}
}
vector<pair<int, pair<int, int>>>all_segment(int n)
{
	vector<pair<int, pair<int, int>>>tmp;
	for (int i = 1; i <= n - 1; i++)
	{
		for (int j = i + 2; j <= n - 1; j++)
		{
			for (int k = j + 2; k <= n - 1 + (i > 0); k++)
			{
				tmp.push_back({ i,{j,k} });
			}
		}
	}
	return tmp;
}
double reverse_segment(vector<int>& tour, int i, int j, int k)
{
	int A = tour[i - 1];
	int B = tour[i];
	int C = tour[j - 1];
	int D = tour[j];
	int E = tour[k - 1];
	int F = tour[k % tour.size()];
	double d0 = get_distance(A, B) + get_distance(C, D) + get_distance(E, F);
	double	d1 = get_distance(A, C) + get_distance(B, D) + get_distance(E, F);
	double	d2 = get_distance(A, B) + get_distance(C, E) + get_distance(D, F);
	double	d3 = get_distance(A, D) + get_distance(E, B) + get_distance(C, F);
	double	d4 = get_distance(F, B) + get_distance(C, D) + get_distance(E, A);
	if (d0 > d1)
	{
		reverse(tour.begin() + i, tour.begin() + j);
		return -d0 + d1;
	}
	else if (d0 > d2)
	{
		reverse(tour.begin() + j, tour.begin() + k);
		return -d0 + d2;
	}
	else if (d0 > d4)
	{
		reverse(tour.begin() + i, tour.begin() + k);
		return -d0 + d4;
	}
	else if (d0 > d3)
	{
		vector<int>tmp;
		for (int ii = j; ii <= k - 1; ii++)tmp.push_back(tour[ii]);
		for (int ii = i; ii <= j - 1; ii++)tmp.push_back(tour[ii]);
		for (int ii = i; ii <= k - 1; ii++)tour[ii] = tmp[ii - i];
		return -d0 + d3;
	}
	return 0;
}
/*
	Get Cluster using KNN
*/
void getCluster()
{
	/*
		Get num_cluster clusters
		Output: vector<vector<int>>cluster
		The clusters including random cities at first
	*/
	int num_cluster = MIN_VEHICLES;
	while (true)
	{
		vector<int>cus;
		for (int i = 0; i <= NUM_OF_CUSTOMERS - 1; i++)
			cus.push_back(i + 1);
		vector<int>clu;
		int step = 0;
		// Init num_cluster root nodes
		while (step < num_cluster)
		{
			//srand(time(0));
			int c = rand() % ((int)cus.size());
			clu.push_back(cus[c]);
			cus.erase(cus.begin() + c);
			step++;
		}
		vector<vector<int>>tmp_cluster(num_cluster);
		vector<int>cap(num_cluster, 0);
		for (int i = 0; i <= num_cluster - 1; i++)
		{
			tmp_cluster[i].push_back(clu[i]);
			cap[i] += get_customer_demand(clu[i]);
		}

		bool f = 1;
		// Sequentially add customers satisfied sum of demmands <= MAX_CAPACITY
		while (cus.size() != 0)
		{
			int cc = *cus.begin();
			int ans = INT_MAX;
			for (int i = 0; i <= num_cluster - 1; i++)
				if (cap[i] + get_customer_demand(cc) <= MAX_CAPACITY)
					ans = min(ans, (int)get_distance(cc, tmp_cluster[i][0]));
			if (ans == INT_MAX)
			{
				f = 0;
				break;
			}
			for (int i = 0; i <= num_cluster - 1; i++)
			{
				bool ok = 0;
				if (cap[i] + get_customer_demand(cc) <= MAX_CAPACITY)
					if ((int)get_distance(cc, tmp_cluster[i][0]) == ans)
					{
						cap[i] += get_customer_demand(cc);
						tmp_cluster[i].push_back(cc);
						cus.erase(cus.begin());
						ok = 1;
						break;
					}
				if (ok)break;
			}
		}
		if (!f)continue;
		else
		{
			cluster = tmp_cluster;
			break;
		}
	}

	//

	for (int i = 0; i <= num_cluster - 1; i++)
		cluster[i].insert(cluster[i].begin(), 0), cluster[i].insert(cluster[i].end(), 0);
}

void getCluster1()
{
	/*
		Get num_cluster clusters
		Output: vector<vector<int>>cluster
		The clusters including random cities at first
	*/
	//cluster.clear();
	int num_cluster = MIN_VEHICLES;
	//while (true)
	//{
	//	vector<int>cus;
	//	for (int i = 0; i <= NUM_OF_CUSTOMERS - 1; i++)
	//		cus.push_back(i + 1);
	//	vector<int>clu;
	//	int step = 0;
	//	// Init num_cluster root nodes
	//	while (step < num_cluster)
	//	{
	//		//srand(time(0));
	//		int c = rand() % ((int)cus.size());
	//		clu.push_back(cus[c]);
	//		cus.erase(cus.begin() + c);
	//		step++;
	//	}
	//	vector<vector<int>>tmp_cluster(num_cluster);
	//	vector<int>cap(num_cluster, 0);
	//	for (int i = 0; i <= num_cluster - 1; i++)
	//	{
	//		tmp_cluster[i].push_back(clu[i]);
	//		cap[i] += get_customer_demand(clu[i]);
	//	}

	//	bool f = 1;
	//	// Sequentially add customers satisfied sum of demmands <= MAX_CAPACITY
	//	while (cus.size() != 0)
	//	{
	//		int cc = *cus.begin();
	//		int ans = INT_MAX;
	//		for (int i = 0; i <= num_cluster - 1; i++)
	//			if (cap[i] + get_customer_demand(cc) <= MAX_CAPACITY)
	//				ans = min(ans, (int)get_distance(cc, tmp_cluster[i][0]));
	//		if (ans == INT_MAX)
	//		{
	//			f = 0;
	//			break;
	//		}
	//		for (int i = 0; i <= num_cluster - 1; i++)
	//		{
	//			bool ok = 0;
	//			if (cap[i] + get_customer_demand(cc) <= MAX_CAPACITY)
	//				if ((int)get_distance(cc, tmp_cluster[i][0]) == ans)
	//				{
	//					cap[i] += get_customer_demand(cc);
	//					tmp_cluster[i].push_back(cc);
	//					cus.erase(cus.begin());
	//					ok = 1;
	//					break;
	//				}
	//			if (ok)break;
	//		}
	//	}
	//	if (!f)continue;
	//	else
	//	{

	//		//vector<int>tmp;
	//		//for (int i = 0; i <= num_cluster - 1; i++)
	//		//{
	//		//	sort(tmp_cluster[i].begin(), tmp_cluster[i].end());
	//		//	for (auto x : tmp_cluster[i])tmp.push_back(x);
	//		//}
	//		//	
	//		//pair<long long, long long>tam = hashh(tmp);
	//		//if (kiemtra.find(tam) == kiemtra.end())
	//		//	kiemtra.insert(tam);
	//		//else
	//		//{
	//		//	continue;
	//		//}
	//		cluster = tmp_cluster;
	//		break;
	//	}
	//}

	//for (int i = 0; i <= num_cluster - 1; i++)
	//	cluster[i].insert(cluster[i].begin(), 0), cluster[i].insert(cluster[i].end(), 0);

	for (int i = 0; i <= num_cluster - 1; i++)
	{
		vector<point>st;
		for (auto cc : cluster[i])
			st.push_back({ cc,node_list[cc].x,node_list[cc].y });
		st.pop_back();
		vector<vector<point>>multiConvex = multiConvexHull(st);
		for (int i = 0; i < multiConvex.size(); i++)
		{
			if (multiConvex[i].size() > 2)
				multiConvex[i].push_back(multiConvex[i][0]);
		}
		hull(multiConvex);
		vector<int>tmp;
		for (auto x : ans)
			tmp.push_back(x.id);
		int idx = 0;
		for (int i = 0; i <= (int)tmp.size() - 1; i++)
			if (tmp[i] == 0)
			{
				idx = i;
				break;
			}
		vector<int>res;
		for (int i = idx; i <= (int)tmp.size() - 2; i++)
			res.push_back(tmp[i]);
		for (int i = 0; i <= (int)idx; i++)
			res.push_back(tmp[i]);
		cluster[i] = res;
	}
}

// EVRP.cpp --------------------
double euclidean_distance(int i, int j) {
	double xd, yd;
	double r = 0.0;
	xd = node_list[i].x - node_list[j].x;
	yd = node_list[i].y - node_list[j].y;
	r = sqrt(xd * xd + yd * yd);
	return r;
}

/*
	Compute the distance matrix of the problem instance
*/
void compute_distances(void) {
	int i, j;
	for (i = 0; i < ACTUAL_PROBLEM_SIZE; i++) {
		for (j = 0; j < ACTUAL_PROBLEM_SIZE; j++) {
			distances[i][j] = euclidean_distance(i, j);
		}
	}
}

/*
	Generate and return a two-dimension array of type double
*/
vector<vector<double>> generate_2D_matrix_double(int n, int m) {
	return vector<vector<double>>(n, vector<double>(m));
}

/*
	Read the problem instance and generate the initial object vector
*/
void read_problem(char* filename) {
	int i;
	char line[CHAR_LEN];
	char* keywords;
	char Delimiters[] = " :=\n\t\r\f\v";
	ifstream fin(filename);
	while ((fin.getline(line, CHAR_LEN - 1))) {

		if (!(keywords = strtok(line, Delimiters)))
			continue;
		if (!strcmp(keywords, "DIMENSION")) {
			if (!sscanf(strtok(NULL, Delimiters), "%d", &problem_size)) {
				cout << "DIMENSION error" << endl;
				exit(0);
			}
		}
		else if (!strcmp(keywords, "EDGE_WEIGHT_TYPE")) {
			char* tempChar;
			if (!(tempChar = strtok(NULL, Delimiters))) {
				cout << "EDGE_WEIGHT_TYPE error" << endl;
				exit(0);
			}
			if (strcmp(tempChar, "EUC_2D")) {
				cout << "not EUC_2D" << endl;
				exit(0);
			}
		}
		else if (!strcmp(keywords, "CAPACITY")) {
			if (!sscanf(strtok(NULL, Delimiters), "%d", &MAX_CAPACITY)) {
				cout << "CAPACITY error" << endl;
				exit(0);
			}
		}
		else if (!strcmp(keywords, "VEHICLES")) {
			if (!sscanf(strtok(NULL, Delimiters), "%d", &MIN_VEHICLES)) {
				cout << "VEHICLES error" << endl;
				exit(0);
			}
		}
		else if (!strcmp(keywords, "ENERGY_CAPACITY")) {
			if (!sscanf(strtok(NULL, Delimiters), "%d", &BATTERY_CAPACITY)) {
				cout << "ENERGY_CAPACITY error" << endl;
				exit(0);
			}
		}
		else if (!strcmp(keywords, "ENERGY_CONSUMPTION")) {
			if (!sscanf(strtok(NULL, Delimiters), "%lf", &energy_consumption)) {
				cout << "ENERGY_CONSUMPTION error" << endl;
				exit(0);
			}
		}
		else if (!strcmp(keywords, "STATIONS")) {
			if (!sscanf(strtok(NULL, Delimiters), "%d", &NUM_OF_STATIONS)) {
				cout << "STATIONS error" << endl;
				exit(0);
			}
		}
		else if (!strcmp(keywords, "OPTIMAL_VALUE")) {
			if (!sscanf(strtok(NULL, Delimiters), "%lf", &OPTIMUM)) {
				cout << "OPTIMAL_VALUE error" << endl;
				exit(0);
			}
		}
		else if (!strcmp(keywords, "NODE_COORD_SECTION")) {
			if (problem_size != 0) {
				/*prroblem_size is the number of customers plus the depot*/
				NUM_OF_CUSTOMERS = problem_size - 1;
				ACTUAL_PROBLEM_SIZE = problem_size + NUM_OF_STATIONS;

				node_list = vector<node>(ACTUAL_PROBLEM_SIZE);

				for (i = 0; i < ACTUAL_PROBLEM_SIZE; i++) {
					//store initial objects
					fin >> node_list[i].id;
					fin >> node_list[i].x >> node_list[i].y;
					node_list[i].id -= 1;
				}
				//compute the distances using initial objects
				distances = generate_2D_matrix_double(ACTUAL_PROBLEM_SIZE, ACTUAL_PROBLEM_SIZE);

			}
			else {
				cout << "wrong problem instance file" << endl;
				exit(1);
			}
		}
		else if (!strcmp(keywords, "DEMAND_SECTION")) {
			if (problem_size != 0) {

				int temp;
				//masked_demand = new int[problem_size];
				//cust_demand = new int[ACTUAL_PROBLEM_SIZE];
				cust_demand = vector<int>(ACTUAL_PROBLEM_SIZE);
				//charging_station = new bool[ACTUAL_PROBLEM_SIZE];
				charging_station = vector<bool>(ACTUAL_PROBLEM_SIZE);
				for (i = 0; i < problem_size; i++) {
					fin >> temp;
					fin >> cust_demand[temp - 1];
				}

				//initialize the charging stations. 
				//the depot is set to a charging station in the DEPOT_SECTION
				for (i = 0; i < ACTUAL_PROBLEM_SIZE; i++) {
					if (i < problem_size) {
						charging_station[i] = false;
					}
					else {
						charging_station[i] = true;
						cust_demand[i] = 0;
					}
				}
			}
		}
		else if (!strcmp(keywords, "DEPOT_SECTION")) {
			fin >> DEPOT;
			DEPOT -= 1;
			charging_station[DEPOT] = true;
		}

	}
	fin.close();
	if (ACTUAL_PROBLEM_SIZE == 0) {
		cout << "wrong problem instance file" << endl;
		exit(1);
	}
	else {
		compute_distances();
	}

}

/*
	 Returns the distance between two points: from and to
*/
double get_distance(int from, int to) {

	return distances[from][to];

}


/*
	Returns the energy consumed when travelling between two points: from and to.
*/
double get_energy_consumption(int from, int to) {

	/*DO NOT USE THIS FUNCTION MAKE ANY CALCULATIONS TO THE ROUTE COST*/
	return energy_consumption * distances[from][to];

}

/*
	Returns the demand for a specific customer
*/
int get_customer_demand(int customer) {

	return cust_demand[customer];

}

/*
	Returns true when a specific node is a charging station; and false otherwise
*/
bool is_charging_station(int node) {

	bool flag = false;
	if (charging_station[node] == true)
		flag = true;
	else
		flag = false;
	return flag;

}