#!/usr/bin/perl -w
use strict;


# my $dollarname = "XXXSEDMEXXX/InputDollarized.inp";
my @sitels = @ARGV;
die "$0: missing arguments" if (scalar(@sitels) == 0);
foreach (@sitels) {$_ = "Cz".$_; }



#my $command = "mkdir -p Plots";
#system($command);

foreach (@sitels) {
	my $target = $_;
	my $dollarname = "$target/InputSzSz.inp";
	my ($nomega, $omega_begin, $omega_step) = getLomega($dollarname);
	$nomega = $nomega - 1;

	my $TPSiteDir = $target;
	$TPSiteDir =~ m/(\d+)/;
	my $TPSite = $1;

	my $output_dyn = "CzDyn.dat";
	my $output_static = "CzStatic.dat";
	open(FILEOUT_DYN, ">", "$output_dyn") or die "$0: Cannot open: $output_dyn: $!\n";
	open(FILEOUT_STATIC, ">", "$output_static") or die "$0: Cannot open: $output_static: $!\n";
	
	foreach (0..$nomega){
		my $nth_omega = $_;
		my $file = "$TPSiteDir/runForinput"."$nth_omega.cout";
		
		my @tempIm;  # for dyn
		my @siteLs_dyn;

		my @Cz;  # for static
		my @sitesLs_static;

		open(FILE, "<", "$file") or die "$0: Cannot open: $file: $!\n";
		while(<FILE>) {
			if (/(\d+)\s\((.*),.*\).*gs.*P1.*\(.*\)/) {
				my $site = $1;
				my $intensity = $2;
				push @sitesLs_static, $site;
				push @Cz, $intensity
				
			}

			if (/(\d+)\s\(.*,(.*)\).*gs.*P2.*\(.*\)/) {
				my $site = $1;
				my $intensity = -$2;
				push @siteLs_dyn, $site;
				push @tempIm, $intensity;
			}
		}
		my $omega = $nth_omega*$omega_step;
		print FILEOUT_DYN "$omega"." "."@tempIm[-18 .. -1]\n";
		print FILEOUT_STATIC "$omega"." "."@Cz[-18 .. -1]\n";
		print "...$nth_omega / $nomega\n";
		close(FILE);
	}
	close(FILEOUT_DYN);
	close(FILEOUT_STATIC);
	
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


