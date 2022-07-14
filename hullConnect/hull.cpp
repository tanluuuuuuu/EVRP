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

typedef long long ll;
typedef unsigned long long ull;
using namespace std;
#define vt vector

struct point
{
	int id;
	float x;
	float y;
	bool add = false;
};

void getAns(vt<vt<point>>, int, int);

float Euclideandistance(point a, point b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
};

vt<point> ans;

void getAns(vt<vt<point>> cvHulls, int index, int id)
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

void hull(vt<vt<point>> convexHulls)
{
	ans.clear();
	/*vt<vt<point>> = {
		{ {1, 0, 0}, {2,3, 0}, {3,4, 4}, {4,-4, 4}, {5,-3, 2}, {1, 0, 0}},
		{ {6, 0, 1}, {7, 1.5, 2.5}, {8,0, 3.5}, {9, -1, 2}, {6, 0, 1} }
	};*/

	ll nConvexHull = convexHulls.size();
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
ll distance2points(point point1, point point2)
{
	return (point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y);
}

/*
	 Find orientation of ordered triplet (point1, point2, point3)
	 0 --> point1, point2 and point3 are collinear
	 1 --> point1, point2 and point3 are clockwise
	 -1 --> point1, point2 and point3 are counterclockwise
*/
//ll orientation3points(point point1, point point2, point point3)
//{
//	ll value_orientation = (point2.y - point1.y) * (point3.x - point2.x) - (point2.x - point1.x) * (point3.y - point2.y);
//	if (value_orientation > 0)
//		return 1;
//	else if (value_orientation < 0)
//		return -1;
//	return 0;
//}

int orientation(point a, point b, point c) {
	double v = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);
	if (v < 0) return -1; // clockwise
	if (v > 0) return +1; // counter-clockwise
	return 0;
}

bool cw(point a, point b, point c, bool include_collinear) {
	int o = orientation(a, b, c);
	return o < 0 || (include_collinear && o == 0);
}

bool ccw(point a, point b, point c, bool include_collinear) {
	int o = orientation(a, b, c);
	return o > 0 || (include_collinear && o == 0);
}

bool collinear(point a, point b, point c) { return orientation(a, b, c) == 0; }

vector<point> GrahamScan(vector<point> a, bool include_collinear = false) {
	if (a.size() == 1)
		return a;

	sort(a.begin(), a.end(), [](point a, point b) {
		return make_pair(a.x, a.y) < make_pair(b.x, b.y);
		});
	point p1 = a[0], p2 = a.back();
	vector<point> up, down;
	up.push_back(p1);
	down.push_back(p1);
	for (int i = 1; i < (int)a.size(); i++) {
		if (i == a.size() - 1 || cw(p1, a[i], p2, include_collinear)) {
			while (up.size() >= 2 && !cw(up[up.size() - 2], up[up.size() - 1], a[i], include_collinear))
				up.pop_back();
			up.push_back(a[i]);
		}
		if (i == a.size() - 1 || ccw(p1, a[i], p2, include_collinear)) {
			while (down.size() >= 2 && !ccw(down[down.size() - 2], down[down.size() - 1], a[i], include_collinear))
				down.pop_back();
			down.push_back(a[i]);
		}
	}

	if (include_collinear && up.size() == a.size()) {
		reverse(a.begin(), a.end());
		return a;
	}
	a.clear();
	for (int i = 0; i < (int)up.size(); i++)
		a.push_back(up[i]);
	for (int i = down.size() - 2; i > 0; i--)
		a.push_back(down[i]);
	return a;
}

// used for sorting points according to polar order w.r.t the pivot
bool POLAR_ORDER(point a, point b) {
	//cout << "Flag 1" << endl;
	ll order = orientation(p0, a, b);
	if (order == 0)
		return distance2points(p0, a) > distance2points(p0, b);
	//cout << "Flag 2" << endl;
	return (order == 1);
}

/*
	Implementation Graham Scan algorithm in order to find the convex hull
*/

void removepoint(vector<point>& points, ll idd)
{
	points.erase(
		remove_if(points.begin(), points.end(), [&](point const& point) {
			return point.id == idd;
			}),
		points.end());
}

vector<vector<point>> multiConvexHull(vector<point> cluster)
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

vector<point> randomGeneratePoints(ll n)
{
	set<pair<ll, ll>> setPoints;
	setPoints.insert({ 0, 0 });

	vt<point> vtpoints;
	vtpoints.push_back({ 1, 0, 0 });

	for (int i = 2; i <= n; i++)
	{
		float xx = rand() % 100;
		float yy = rand() % 100;
		while (setPoints.find({ xx, yy }) != setPoints.end())
		{
			xx = rand() % 100;
			yy = rand() % 100;
		}
		vtpoints.push_back({ i, xx, yy });
	}
	//for (int i = 0; i < n; i++)
	//{
	//	cout << vtpoints[i].id << " " << vtpoints[i].x << " " << vtpoints[i].y << endl;
	//}
	return vtpoints;
}

//---------------------------------

int main()
{
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
	//hull();

	while (true)
	{
		ll n;
		n = rand() % 101 + 3;
		vector<point>st = randomGeneratePoints(n);

		vector<vector<point>> multiConvex = multiConvexHull(st);

		for (int i = 0; i < multiConvex.size(); i++)
		{
			if (multiConvex[i].size() > 2)
				multiConvex[i].push_back(multiConvex[i][0]);
		}
		hull(multiConvex);

		if (ans.size() != n + 1 || ans[0].id != ans[ans.size() - 1].id)
		{
			cout << "DIFF" << endl;
			cout << "n: " << n << endl;
			cout << "ans: " << ans.size() << endl;
			cout << "[";
			for (ll i = 0; i < multiConvex.size(); i++)
			{
				//cout << "Polygon " << i + 1 << " - " << multiConvex[i].size() << ": ";
				cout << "\"";
				for (ll j = 0; j < multiConvex[i].size() - 1; j++)
				{
					cout << multiConvex[i][j].id << " ";
				}
				cout << multiConvex[i][multiConvex[i].size() - 1].id;
				cout << "\", ";
			}
			cout << "]";

			for (auto x : ans)
			{
				cout << x.id << " ";
			}
			system("pause");
		}
		else if (ans[0].id == ans[1].id)
		{
			cout << "overlap" << endl;
			for (auto x : ans)
			{
				cout << x.id << " ";
			}
			system("pause");
		}
		else
		{
			cout << "OK" << endl;
			/*for (auto x : ans)
			{
				cout << x.id << " ";
			}
			system("pause");*/
		}
	}

	vt<point> st;
	st.push_back({ 1, 0, 0 });
	st.push_back({ 2, 93, 82 });
	st.push_back({ 3, 71, 69 });
	st.push_back({ 4, 93, 77 });
	st.push_back({ 5, 14, 4 });
	st.push_back({ 6, 84, 43 });
	st.push_back({ 7, 95, 11 });
	st.push_back({ 8, 93, 92 });
	st.push_back({ 9, 98, 90 });
	st.push_back({ 10, 99, 90 });
	st.push_back({ 11, 93, 74 });

	vector<vector<point>>multiConvex = multiConvexHull(st);

	for (int i = 0; i < multiConvex.size(); i++)
	{
		if (multiConvex[i].size() > 2)
			multiConvex[i].push_back(multiConvex[i][0]);
	}
	hull(multiConvex);
	//double sumDis = 0;
	//for (int i = 0; i < ans.size() - 1; i++)
	//{
	//	sumDis += Euclideandistance(ans[i], ans[i + 1]);
	//}

	for (auto x : ans)
	{
		cout << x.id << " ";
	}
	//cout << endl;
	//cout << "TOTAL Dis: " << sumDis << endl;

	//cout << "[";
	//for (ll i = 0; i < multiConvex.size(); i++)
	//{
	//	//cout << "Polygon " << i + 1 << " - " << multiConvex[i].size() << ": ";
	//	cout << "\"";
	//	for (ll j = 0; j < multiConvex[i].size() - 1; j++)
	//	{
	//		cout << multiConvex[i][j].id << " ";
	//	}
	//	cout << multiConvex[i][multiConvex[i].size() - 1].id;
	//	cout << "\", ";
	//}
	//cout << "]";

	//for (ll i = 0; i < multiConvex.size(); i++)
	//{
	//	cout << "Polygon " << i + 1 << " - " << multiConvex[i].size() << ": ";
	//	for (ll j = 0; j < multiConvex[i].size(); j++)
	//	{
	//		cout << "(" << multiConvex[i][j].id << " " << multiConvex[i][j].x << " " << multiConvex[i][j].y << ")" << " ";
	//	}
	//	cout << "\n";
	//}
}