package com.model;

import java.util.Hashtable;

import com.control.PayOff;
import com.tools.Value;


public class Model {

	//////////////////////////////////////////////////////////////////
	// Globale variable
	/////////////////////////////////////////////////////////////////
	/*********************************************************************/


//	final public static String DEFAULT_PATH_CAMPAIGN = "/home/oishi/CAMPAIGN";
//	final public static String DEFAULT_CAMPAIGN_FOLDER = "HITZ";
//	final public static String DEFAULT_PATH_INP = DEFAULT_PATH_CAMPAIGN+"/"+DEFAULT_CAMPAIGN_FOLDER+"/INP";
//	final public static String DEFAULT_PATH_RNX = DEFAULT_PATH_CAMPAIGN+"/"+DEFAULT_CAMPAIGN_FOLDER+"/RNX";
//	final public static String DEFAULT_PATH_STA = DEFAULT_PATH_CAMPAIGN+"/"+DEFAULT_CAMPAIGN_FOLDER+"/STA";
//	final public static String DEFAULT_PATH_GEN = DEFAULT_PATH_CAMPAIGN+"/"+DEFAULT_CAMPAIGN_FOLDER+"/GEN";
//	final public static String DEFAULT_PATH_ORB = DEFAULT_PATH_CAMPAIGN+"/"+DEFAULT_CAMPAIGN_FOLDER+"/ORB";
//	final public static String DEFAULT_PATH_OUT = DEFAULT_PATH_CAMPAIGN+"/"+DEFAULT_CAMPAIGN_FOLDER+"/OUT";
//	final public static String DEFAULT_PATH_ATM = DEFAULT_PATH_CAMPAIGN+"/"+DEFAULT_CAMPAIGN_FOLDER+"/ATM";
	
	
	///////////////////////////////////////////////////////////////
	// Declaration of the value
	//////////////////////////////////////////////////////////////

//	private Value vPathFolderCAMPAIGN = new Value(DEFAULT_PATH_CAMPAIGN);
//	private Value vFolderCAMPAIGN = new Value(DEFAULT_CAMPAIGN_FOLDER);
//	private Value vPathFolderINP = new Value(DEFAULT_PATH_INP);
//	private Value vPathFolderRNX = new Value(DEFAULT_PATH_RNX);
//	private Value vPathFolderSTA = new Value(DEFAULT_PATH_STA);
//	private Value vPathFolderGEN = new Value(DEFAULT_PATH_GEN);
//	private Value vPathFolderORB = new Value(DEFAULT_PATH_ORB);
//	private Value vPathFolderOUT = new Value(DEFAULT_PATH_OUT);
	
	private Value vPayOff = new Value(new PayOff());
	private Value vTheory = new Value(false);
	private Value vAttachment = new Value(false);
	
//	/////////////////////////////////////////////////////////////////////
//	//  Hash table which contain all the parameter
//	// an unique identifiant is choose for each variable
//	/////////////////////////////////////////////////////////////////////
	private Hashtable<String, Object> hTable = new Hashtable<String, Object>();

	/////////////////////////////////////////////////////////////////////
	//  Hash table which contain all the ALGORITHM PARAMETERS
	// an unique identifiant is choose for each variable
	/////////////////////////////////////////////////////////////////////


	public Model() {

		//-------------------------------------------------
//		Properties prop = System.getProperties();
//		
//		prop.put("https.proxySet", "true");	
//		prop.put("http.proxyHost", "192.168.1.2");
//		prop.put("http.proxyPort", "3128");		
//		prop.put("https.proxyHost", "192.168.1.2");
//		prop.put("https.proxyPort", "3128");
		//-------------------------------------------------
		
		

		
		hTable.put("PayOff",vPayOff);
		hTable.put("Theory",vTheory);
		hTable.put("Attachment",vAttachment);
//		hTable.put("PathFolderRNX",vPathFolderRNX);
//		hTable.put("PathFolderSTA",vPathFolderSTA);
//		hTable.put("PathFolderGEN",vPathFolderGEN);
//		hTable.put("PathFolderORB",vPathFolderORB);
//		hTable.put("PathFolderOUT",vPathFolderOUT);
//		hTable.put("PathFolderATM",vPathFolderATM);	
//		hTable.put("PathFolderRAW",vPathFolderRAW);



	}

	/**
	 * return the good value, according to the name
	 * @return Value
	 */
	public Value getValue(String sKey) {
		return (Value) hTable.get(sKey);
	}


	/**
	 * return the content of value giving in parameters, according to the name
	 * @return Value
	 */
	public Object getValueObj(String sKey) {
		return ((Value)hTable.get(sKey)).getValue();
	}


	/**
	 * get the hashtable values
	 * @return Hashtable
	 */
	public Hashtable getHashTable() {
		return hTable;
	}

}

