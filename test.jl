using GeneticVariation
using DataFrames
using Logging
using Distributions
using CodecZlib

include("utils.jl")
include("firth.jl")
Logging.configure(level=INFO)

function main()
    # const test_file = "../data/test.vcf"
    const test_file = "../data/MGI_HRC_chr21.dose.vcf.gz"
    const ped_file = "../data/MGI.filtered.discrete.FINAL.ped"
    reader = VCF.Reader(GzipDecompressionStream(open(test_file, "r"))) 
    info("parsing PED file now")
    ped = readtable(ped_file, separator = '\t', normalizenames = false)
        
    # diabetes
    phenotype = ped[Symbol("250.2")]
    trimmed_phenotype = trim_and_convert_phenotype(phenotype) 
    trimmed_phenotype = trimmed_phenotype .- 1
    const vcf_ids = header(reader).sampleID
    covars = extract_covars(ped, phenotype)

    const valid_ped_ids = return_valid_ped_ids(phenotype, ped[:IND_ID])
    const vcf_indices = return_valid_vcf_ids(valid_ped_ids, vcf_ids)

    info("now parsing VCF")
    count = 0
    genotype = Float64[]
    result = DataFrame(
                    BETA = Float64[], 
                    SEBETA = Float64[], 
                    CHISQ = Float64[], 
                    PVALUE = Float64[],
                    CHROM = String[],
                    BEGIN = Int64[],
                    END = Int64[],
                    MARKER_ID = String[],
                    NS = Int64[],
                    AC = Float64[],
                    MAF = Float64[]
                )
    covars[:genotype] = ones(size(covars, 1))
    covars = convert(Matrix, covars)
    covars = map(y->convert(Float64, y), covars)
    for record in reader
       #  if count < 1000 
            MARKER_ID = @sprintf(
                            "%s:%d_%s:%s_%s:%d", 
                            VCF.chrom(record), 
                            VCF.pos(record), 
                            VCF.ref(record), 
                            VCF.alt(record)[1], 
                            VCF.chrom(record), 
                            VCF.pos(record)
                        )::String  
            genotype = VCF.genotype(record, vcf_indices, "DS")
            genotype = map(x->parse(Float64, x), genotype)
            covars[:, size(covars, 2)] = genotype
            CHROM = VCF.chrom(record)::String
            POS = VCF.pos(record)::Int64
            NS = size(covars, 1)::Int64
            AC = sum(genotype)::Float64
            MAF = AC / (2 * NS)
            if MAF >= 0.01
                info("running logit now: ", count)
                res = single_b_firth(covars, trimmed_phenotype, false)
                res[:CHROM] = CHROM
                res[:BEGIN] = POS
                res[:END] = POS
                res[:MARKER_ID] = MARKER_ID
                res[:NS] = NS
                res[:AC] = AC
                res[:MAF] = MAF
                append!(result, res)
                info("now done with logit")
            else
                info("skipping number", count, " ", MARKER_ID, " because MAF < 0.01")
            end
      #  end
        count = count + 1
    end
    info("now done with all tests")
    return result[[:CHROM, 
                   :BEGIN,
                   :END,
                   :MARKER_ID,
                   :NS,
                   :AC,
                   :MAF,
                   :BETA,
                   :SEBETA,
                   :CHISQ,
                   :PVALUE]]
end

#main()
