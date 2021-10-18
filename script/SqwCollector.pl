#!/usr/bin/perl -w
use strict;


# my $dollarname = "XXXSEDMEXXX/InputDollarized.inp";
my @sitels = @ARGV;
die "$0: missing arguments" if (scalar(@sitels) == 0);
foreach (@sitels) {$_ = "SZ".$_; }



#my $command = "mkdir -p Plots";
#system($command);

foreach (@sitels) {
	my $target = $_;
	my $dollarname = "$target/InputDollarized.inp";
	my ($nomega, $omega_begin, $omega_step) = getLomega($dollarname);
	$nomega = $nomega - 1;

	my $TPSiteDir = $target;
	$TPSiteDir =~ m/(\d+)/;
	my $TPSite = $1;

	my $output = "$TPSiteDir".".DOS.dat";
	open(FILEOUT, ">", "$output") or die "$0: Cannot open: $output: $!\n";
	foreach (0..$nomega){
		my $nth_omega = $_;
		my $file = "$TPSiteDir/runForinput"."$nth_omega.cout";
		
		my @tempIm;
		my @siteLs;

		open(FILE, "<", "$file") or die "$0: Cannot open: $file: $!\n";
		while(<FILE>) {
			if (/(\d+)\s\(.*,(.*)\).*gs.*P2.*\(.*\)/) {
				my $site = $1;
				my $intensity = -$2;
				#$intensity = -1*$intensity;
				#if ($intensity < 0){
				#	$intensity = 0;
				#}
				push @siteLs, $site;
				push @tempIm, $intensity;
			}
		}
		my $omega = $nth_omega*$omega_step;
		print FILEOUT "$omega"." "."@tempIm[-32 .. -1]\n";
		print "...$nth_omega / $nomega\n";
		close(FILE);
	}
	close(FILEOUT);
	
}



sub getLomega 
{
	my ($file) = @_;
	my @omegalist;
	my $nomega;
	my $omega_step;
	my $omega_begin;
	open(FILE, "<", $file) or die "$0: Cannot open$file: $!\n";
	while(<FILE>) {
		if (/^[\s]*#OmegaTotal/) {
			my $line = $_;
			$line =~ m/(\d+)/;
			$nomega = $1;
			die "$0: I cannot match in: $line \n" if ($nomega eq "");
		} elsif (/^[\s]*#OmegaStep/) {
			my $line = $_;
			$line =~ m/(\d+.\d+)/;
			$omega_step = $1;	
			die "$0: I cannot match in: $line \n" if ($omega_step eq "");
		} elsif (/^[\s]*#OmegaBegin/) {
			my $line = $_;
			$line =~ m/(\d+)/;
			$omega_begin = $1;
			die "$0: I cannot match in: $line \n" if ($omega_begin eq "");
		}

	}
	close(FILE);
	push @omegalist, ($nomega, $omega_begin, $omega_step);
	print "Nomega=$nomega, OmegaBegin=$omega_begin, OmegaStep=$omega_step\n";
	return @omegalist;
}


