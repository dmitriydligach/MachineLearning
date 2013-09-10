#!/usr/bin/perl

use strict;

($#ARGV > 0) or die("Required arguments: <curve 1> [starting point]\n");

open FP, " < $ARGV[0]";

my $starting_point = -1;

if($#ARGV == 1){ $starting_point = $ARGV[1]; }
my $last_f1;
my $last_ind;
my $alc = 0.0; ## Area under the Learning Curve

while(<FP>){
    if(m/^(\d+) ([\d\.]+)(.*)/){
        my $inst_num = $1;
        my $f1 = $2;
        my $fold_std = $3;
        $fold_std =~ s/^ *//;
        if($starting_point == -1){
            $starting_point = $inst_num;
        }elsif($starting_point != $inst_num){
            my $low, my $high;
            if($f1 < $last_f1){
                $low = $f1;
                $high = $last_f1;
            }else{
                $low = $last_f1;
                $high = $f1;
            }
            $alc += $low * ($inst_num - $last_ind) + 0.5 * ($high-$low) * ($inst_num-$last_ind);
            #print "Running ALC: $inst_num $alc\n";
        }else{
            ## this is the first point, we can ignore it
        }
        $last_ind = $inst_num;
        $last_f1 = $f1;
    }    
}

close FP;

print "ALC = $alc\n";
