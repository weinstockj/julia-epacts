using GeneticVariation
using DataFrames
using Logging

include("utils.jl")
include("firth.jl")
Logging.configure(level=INFO)

function main()
    test_file = "../data/test.vcf"
    ped_file = "../data/MGI.filtered.discrete.FINAL.ped"
    reader = VCF.Reader(open(test_file, "r")) 
    info("parsing PED file now")
    ped = readtable(ped_file, separator = '\t', normalizenames = false)
        
    # diabetes
    phenotype = ped[Symbol("250.2")]
    trimmed_phenotype = trim_and_convert_phenotype(phenotype) 
    trimmed_phenotype = trimmed_phenotype .- 1
    ped_ids = ped[:IND_ID]
    vcf_ids = header(reader).sampleID
    covars = extract_covars(ped, phenotype)

    valid_ped_ids = return_valid_ped_ids(phenotype, ped_ids)
    vcf_indices = return_valid_vcf_ids(valid_ped_ids, vcf_ids)

    info("now parsing VCF")
    count = 0
    genotype = []
    for record in reader
        if count < 1
            genotype = VCF.genotype(record, vcf_indices, "DS")
            genotype = map(x->parse(Float64, x), genotype)
            covars[:genotype] = genotype
            x = convert(Matrix, covars)
            x = map(y->convert(Float64, y), x)
            info("running logit now")
            res = fast_logistf_fit(x, trimmed_phenotype)
            info("now done with logit")
        end
        count = count + 1
    end
end

#main()
