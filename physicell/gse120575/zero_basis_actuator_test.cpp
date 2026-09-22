#include <cmath>
#include <iostream>
#include <vector>

static int rounded_with_residual(double requested,double& residual){
    double due=requested+residual;
    int scheduled=due>=0?int(std::floor(due+0.5)):int(std::ceil(due-0.5));
    residual=due-scheduled;
    return scheduled;
}

int main(){
    const int initial_type_workers=0,initial_total_workers=600;
    std::vector<double> budgets={0,0.001,0.002,0.005,0.01};
    std::cout<<"initial_type_workers,initial_total_workers,external_recruitment_budget,recruitment_enabled,interval,requested_recruitment,scheduled_recruitment,residual_after,end_type_workers,false_emergence,event_balance\n";
    for(double budget:budgets){
        for(int enabled_int=0;enabled_int<=1;++enabled_int){
            bool enabled=enabled_int==1;double residual=0;int current=initial_type_workers;
            for(int interval=0;interval<5;++interval){
                int start=current;double requested=enabled?budget*initial_total_workers:0.0;
                int scheduled=rounded_with_residual(requested,residual);
                int recruited=std::max(0,scheduled);current+=recruited;
                int false_emergence=(!enabled||budget==0)&&current!=0;
                int balance=(current-start)-recruited;
                std::cout<<initial_type_workers<<","<<initial_total_workers<<","<<budget<<","<<(enabled?"true":"false")<<","<<interval<<","<<requested<<","<<recruited<<","<<residual<<","<<current<<","<<false_emergence<<","<<balance<<"\n";
            }
        }
    }
}
