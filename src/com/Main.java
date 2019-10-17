package com;

import com.control.PayOff;
import com.control.Tournament;
import com.control.TournamentResult;
import com.gui.Gui;
import com.model.Model;

public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		Model model = new Model();
		
		new Gui(model);
		
		
		
//		int[][] payoff = ((PayOff) model.getValue("PayOff").getValue()).getPayOff(); 
//		TournamentResult[][][][] totalResult = new TournamentResult[10][10][10][10];
//		Tournament tournament = new Tournament(model); 
//		payoff[0][0]=3;
//		payoff[0][1]=7;
//		payoff[1][0]=8;
//		payoff[1][1]=4;
//		tournament.runTournament();
//		tournament.showResult();
		
		
//		A100 : 8568
//		Generous : 7248
//		Greedy : 5920
//		BeatLast : 4736
//		TitForTat : 2844
//		Random : 1758
//		MinMax : 635
//		B100 : 630
		
//		for (int i = 0; i < totalResult.length; i++) {
//			TournamentResult[][][] tournamentResults = totalResult[i];
//			for (int j = 0; j < tournamentResults.length; j++) {
//				TournamentResult[][] tournamentResults2 = tournamentResults[j];
//				for (int k = 0; k < tournamentResults2.length; k++) {
//					TournamentResult[] tournamentResults3 = tournamentResults2[k];
//					for (int l = 0; l < tournamentResults3.length; l++) {
//						payoff[0][0]=i+1;
//						payoff[0][1]=j+1;
//						payoff[1][0]=k+1;
//						payoff[1][1]=l+1;
//						
//						
//						tournament.clearResults();
//						for (int l2 = 0; l2 < 100; l2++) {
//							tournament.runTournament();							
//						}
//						
//						tournamentResults3[l]=tournament.getResult();
//						tournamentResults3[l].sort();
//						
//						System.out.println((i+1)+" "+(j+1)+" "+(k+1)+" "+(l+1)+" :"+ tournamentResults3[l].getListResultat().get(0).getName()+" "+tournamentResults3[l].getListResultat().get(0).getResult());
//					}
//				}
//			}
//		}
		
		
		
//		for (int i = 0; i < payoff.length; i++) {
//			for (int j = 0; j < payoff[i].length; j++) {
//				payoff[i][j] = (int)(10*Math.random()+1);
//				System.out.print(payoff[i][j]+" ");
//			}
//			System.out.println();
//		}
//		
//		Tournament tournament = new Tournament(model); 
//		tournament.clearResults();
//		for (int i = 0; i < 100; i++) {
//		tournament.runTournament();
//		}
//		
//		
//		tournament.showResult();
		
		
//		new Gui(model);
	}

	
	
}
