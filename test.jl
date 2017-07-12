using GeneticVariation
using DataFrames
using Logging

include("utils.jl")
Logging.configure(level=INFO)

test_file = "../data/test.vcf"
ped_file = "../data/MGI.filtered.discrete.FINAL.ped"
reader = VCF.Reader(open(test_file, "r")) 
info("parsing PED file now")
ped = readtable(ped_file, separator = '\t', normalizenames = false)

    
# diabetes
phenotype = ped[Symbol("250.2")]
ped_ids = ped[:IND_ID]
vcf_ids = header(reader).sampleID

valid_ped_ids = return_valid_ped_ids(phenotype, ped_ids)
vcf_indices = return_valid_vcf_ids(valid_ped_ids, vcf_ids)

info("now parsing VCF")
count = 0
genotype = []
for record in reader
    if count < 4
        genotype = VCF.genotype(record, vcf_indices, "DS")
        genotype = map(x->parse(Float64, x), genotype)
    end
    count = count + 1
end

