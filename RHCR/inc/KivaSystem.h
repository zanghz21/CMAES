#pragma once
#include "BasicSystem.h"
#include "KivaGraph.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class KivaSystem :
	public BasicSystem
{
public:
	KivaSystem(const KivaGrid& G, MAPFSolver& solver);
	~KivaSystem();

	json simulate(int simulation_time);
	json summarizeResult();
	json summarizeCurrResult(int summarize_interval, bool congested_sim);
	json warmup(int warmup_time);
	json update_gg_and_step(int update_gg_interval);
	int total_sim_time;
	void set_total_sim_time(int total_sim_time, int warmup_time);


private:
	const KivaGrid& G;
	unordered_set<int> held_endpoints;
	std::vector<string> next_goal_type;

	void initialize();
	void initialize_start_locations();
	void initialize_goal_locations();
	void update_goal_locations();
	int gen_next_goal(int agent_id, bool repeat_last_goal=false);
    int sample_workstation();
    tuple<vector<double>, double, double> edge_pair_usage_mean_std(
        vector<vector<double>> &edge_usage);
    tuple<vector<vector<vector<double>>>, vector<vector<double>>> convert_edge_usage(vector<vector<double>> &edge_usage);

    // Used for workstation sampling
    discrete_distribution<int> workstation_dist;
    mt19937 gen;
};

