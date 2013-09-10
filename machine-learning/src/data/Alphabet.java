package data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Map strings to integers and back.
 * TODO: Perhaps "vocabulary" is a better name for this class.
 * 
 * @author dmitriy dligach
 *
 */
public class Alphabet {

	// start indexing from this value
	// looks like libsvm is fine with this too
	public static final int startIndex = 0; 
	
	private Map<String, Integer> str2int;
	private Map<Integer, String> int2str;
	private int index;
	
	public Alphabet() {
		str2int  = new HashMap<String, Integer>();
		int2str = new HashMap<Integer, String>();
		index = startIndex;
	}

	public void add(String str) {
		
		if(! str2int.containsKey(str)) {
			str2int.put(str, index);
			int2str.put(index, str);
			index++;
		}
	}

	public int size() {
		return str2int.size();
	}
	
	public int getIndex(String str) {
		return str2int.get(str); 
	}
	
	public String getString(int index) {
		return int2str.get(index);
	}

	/**
	 * Get strings in the order of increasing indexes.
	 */
	public List<String> getStrings() {

		List<String> strings = new ArrayList<String>();
		
		for(int i = startIndex; i < size(); i++) {
			strings.add(int2str.get(i));
		}
		
		return strings;
	}
	
	public String toString() {
		return str2int.toString();
	}
}
