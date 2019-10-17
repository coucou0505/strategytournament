package com.control;

import com.control.strategy.AbstractStrategy;
import com.control.strategy.StrategySet;
import com.model.Model;

public class Tournament {

	private Model model;
	private StrategySet strategySet=null;
	private int numTournaments = 0;
	private TournamentResult results=null;

	public Tournament(Model model) {
		this.model = model;
		strategySet = new StrategySet(model);
		clearResults();
	}

	public void clearResults(){
		numTournaments = 0;
		results = new TournamentResult(strategySet);
	}


	public void runTournament(){
		PayOff payOff = (PayOff)model.getValue("PayOff").getValue();
		AbstractStrategy vStrat;
		AbstractStrategy hStrat;
		ResultatStrategy restTemp;

		int hLast = 1;
		int vLast = 1;

		int hMove;
		int vMove;

		int winnerScore =0;
		int placeScore =0;
		int showScore =0;
		int beatBoost = 0;
		int oldAverage;

		int[] scores = new int[strategySet.getStrategyNumber()];
		int[] yomi = new int[strategySet.getStrategyNumber()];

		boolean has_theory;
		boolean has_attachment;
	
		
		for (int i = 0; i < strategySet.getStrategyNumber(); i++) {
			vStrat = strategySet.getStrategy(i);

			for (int j = 0; j < strategySet.getStrategyNumber(); j++) {
				hStrat = strategySet.getStrategy(j);



				for (int k = 0; k < 12; k++) {

					vMove = vStrat.action(hLast);
					hMove = hStrat.action(vLast);

					vLast = vMove;
					hLast = hMove;

					if(vMove == 1 && hMove == 1) {
						scores[i] += payOff.getAA();
						scores[j] += payOff.getAA();
					} else if(vMove == 1 && hMove == 2) {
						scores[i] += payOff.getAB();
						scores[j] += payOff.getBA();
					} else if(vMove == 2 && hMove == 1) {
						scores[i] += payOff.getBA();
						scores[j] += payOff.getAB();
					} else {
						scores[i] += payOff.getBB();
						scores[j] += payOff.getBB();
					}


				}

			}
		}
		

		// Strategic attachment treats tied scores as only taking
		// a single spot, so e.g. 5 5 5 4 4 3 3 2 would give
		// the second place bonus to both 4s and the third place bonus
		// to both 3s.
		for(int j = 0; j < scores.length; j++) {
			if(scores[j] > winnerScore) {
				winnerScore = scores[j];
			}
		}

		for(int j = 0; j < scores.length; j++) {
			if(scores[j] < winnerScore && scores[j] > placeScore) {
				placeScore = scores[j];
			}
		}

		for(int j = 0; j < scores.length; j++) {
			if(scores[j] < placeScore && scores[j] > showScore) {
				showScore = scores[j];
			}
		}

		for(int i = 0; i < scores.length; i++) {
			// Compute the yomi gains for each strategy.
			// The basic yomi gain formula is points*beatBoost
			// where beatBoost is the number of strats beaten (or 1
			// if the strategy got last). However, ties are not treated
			// equally. The game creates a `results` array using a
			// stable selection sort, and the position of a strat in
			// that array determines how many strats it beat, so
			// in the case of ties the earlier strat in the list
			// generates more yomi.
			
			beatBoost = 0;
			for(int j = 0; j < scores.length; j++) {
				if(scores[i] > scores[j] || (i < j && scores[i] == scores[j])) {
					beatBoost += 1;
				}
			}
			if(beatBoost == 0) {
				beatBoost = 1;
			}

			has_theory = (Boolean) model.getValue("Theory").getValue();

			if(has_theory) {
				yomi[i] = scores[i] * beatBoost * 2;
			} else {
				yomi[i] = scores[i] * beatBoost;
			}

			has_attachment = (Boolean) model.getValue("Attachment").getValue();

			if(has_attachment) {
				if(scores[i] == winnerScore) {
					yomi[i] += 50000;
				} else if(scores[i] == placeScore) {
					yomi[i] += 30000;
				} else if(scores[i] == showScore) {
					yomi[i] += 20000;
				}
			}
		}

		numTournaments ++;

		for(int i = 0; i < strategySet.getStrategyNumber(); i++) {
			restTemp = results.getResult(strategySet.getStrategy(i).getName());
			oldAverage = restTemp.getResult();
			restTemp.setResult( (oldAverage * (numTournaments - 1) + yomi[i]) / numTournaments);
		}

	}

	
	public TournamentResult getResult(){
		return results;
	}
	
	public String getResultString(){
		String result = new String();
		ResultatStrategy res;
		results.sort();
		
		
		for (int i = 0; i < results.getListResultat().size(); i++) {
			
			res = results.getListResultat().get(i);
			result = result + (res.getName()+" : "+res.getResult()+"\n");
		}
		
		return result;
	}
	
	public void showResult(){
		results.sort();
		ResultatStrategy res;
		
		for (int i = 0; i < results.getListResultat().size(); i++) {
			
			res = results.getListResultat().get(i);
			System.out.println(res.getName()+" : "+res.getResult());
		}
		System.out.println();
	}
}
