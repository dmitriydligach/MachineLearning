package utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Read a file mapping CUIs to UMLS preferred terms.
 * This file may contain multiple entries for the same CUI.
 * Read only the first one and ignore the rest.
 */
public class UmlsLookup {
  
  Map<String, String> cui2text = new HashMap<String, String>();
  
  public UmlsLookup(String file) throws IOException {
    
    String line;
    BufferedReader reader = new BufferedReader(new FileReader(file));
    while((line = reader.readLine()) != null){
      String[] fields = line.split("\\|");
      String cui = fields[0];
      String umlsPreferredTerm = fields[2];
      if(cui2text.containsKey(cui)) {
        continue; 
      } else {
        cui2text.put(cui, umlsPreferredTerm);
      }
    }
    reader.close();
  }
  
  public String getTerm(String cui){
    return cui2text.get(cui);
  }

  public static void main(String[] args) throws IOException {
    
    UmlsLookup lookup = new UmlsLookup("/Users/dima/Boston/Data/Umls/rxnorm-snomedct.txt");
    String text = lookup.getTerm("C0022671");
    System.out.println("snomedct entry: " + text);
    System.out.println("rxnorm entry: " + lookup.getTerm("C2343521"));
  }
}
